import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight


class MambaBlock(nn.Module):
    """
    Single Mamba block: norm → in_proj → [x, z] → conv1d → SiLU → SSM → gate → out_proj + residual

    Supports two modes:
    - Sequential (training): full sequence in, full sequence out
    - Recurrent (streaming): one timestep at a time with carried hidden state
    """

    def __init__(self, d_model, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.d_inner = d_model * expand
        self.dt_rank = math.ceil(d_model / 16)

        self.norm = RMSNorm(d_model)

        # Input projection: d_model → 2 * d_inner (split into x and z branches)
        self.in_proj = nn.Linear(d_model, 2 * self.d_inner, bias=False)

        # Causal depthwise conv on x branch
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=d_conv - 1,
            groups=self.d_inner, bias=True
        )

        # SSM input-dependent projections from x
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * d_state, bias=False)

        # dt projection: dt_rank → d_inner
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # SSM parameters
        # A: structured as log for numerical stability, initialized with -log(1..d_state)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).expand(self.d_inner, -1)
        self.A_log = nn.Parameter(torch.log(A))

        # D: skip connection parameter
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _ssm_step(self, x_t, h, dt, B, C):
        """Single recurrent SSM step.

        Args:
            x_t: (B, d_inner) - input at this timestep
            h:   (B, d_inner, d_state) - previous hidden state
            dt:  (B, d_inner) - discretization step
            B:   (B, d_state) - input-dependent B
            C:   (B, d_state) - input-dependent C

        Returns:
            y_t: (B, d_inner) - output
            h:   (B, d_inner, d_state) - new hidden state
        """
        A = -torch.exp(self.A_log)  # (d_inner, d_state)
        dt = F.softplus(dt)  # (B, d_inner)

        # Discretize
        A_bar = torch.exp(dt.unsqueeze(-1) * A)  # (B, d_inner, d_state)
        B_bar = dt.unsqueeze(-1) * B.unsqueeze(1)  # (B, d_inner, d_state)

        # Recurrence
        h = A_bar * h + B_bar * x_t.unsqueeze(-1)  # (B, d_inner, d_state)

        # Output
        y_t = (h * C.unsqueeze(1)).sum(-1) + self.D * x_t  # (B, d_inner)

        return y_t, h

    def _ssm_sequential(self, x, conv_state=None, ssm_state=None):
        """Process full sequence through SSM (training mode).

        Args:
            x: (B, L, d_inner) - post-conv, post-SiLU input
            conv_state: not used in sequential mode
            ssm_state: (B, d_inner, d_state) or None

        Returns:
            y: (B, L, d_inner) - SSM output
            final_ssm_state: (B, d_inner, d_state)
        """
        batch, seq_len, _ = x.shape

        # Project to get dt, B, C for all timesteps
        x_dbc = self.x_proj(x)  # (B, L, dt_rank + 2*d_state)
        dt_raw, B, C = torch.split(x_dbc, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt_raw)  # (B, L, d_inner)

        # Initialize hidden state
        if ssm_state is None:
            h = torch.zeros(batch, self.d_inner, self.d_state, device=x.device, dtype=x.dtype)
        else:
            h = ssm_state

        # Sequential scan (pure PyTorch - no custom CUDA kernel needed)
        outputs = []
        for t in range(seq_len):
            y_t, h = self._ssm_step(x[:, t], h, dt[:, t], B[:, t], C[:, t])
            outputs.append(y_t)

        y = torch.stack(outputs, dim=1)  # (B, L, d_inner)
        return y, h

    def forward(self, x, state=None):
        """
        Args:
            x: (B, L, d_model) input features
            state: dict with 'conv_state' and 'ssm_state', or None

        Returns:
            output: (B, L, d_model)
            new_state: dict with 'conv_state' and 'ssm_state'
        """
        batch, seq_len, _ = x.shape
        residual = x

        # Norm
        x = self.norm(x)

        # Input projection → split into x and z branches
        xz = self.in_proj(x)  # (B, L, 2*d_inner)
        x_branch, z = xz.chunk(2, dim=-1)  # each (B, L, d_inner)

        # Causal conv1d on x branch
        if state is not None and state.get('conv_state') is not None:
            # Recurrent mode: prepend conv state for causal context
            conv_input = torch.cat([state['conv_state'], x_branch], dim=1)  # (B, d_conv-1+L, d_inner)
            conv_input = conv_input.transpose(1, 2)  # (B, d_inner, d_conv-1+L)
            x_conv = self.conv1d(conv_input)[:, :, :seq_len]  # trim padding, take last L
            # Save new conv state (last d_conv-1 timesteps of x_branch)
            new_conv_state = x_branch[:, -(self.d_conv - 1):, :].detach()
        else:
            # Sequential mode: standard causal conv
            x_branch_t = x_branch.transpose(1, 2)  # (B, d_inner, L)
            x_conv = self.conv1d(x_branch_t)[:, :, :seq_len]  # causal trim
            new_conv_state = x_branch[:, -(self.d_conv - 1):, :].detach()

        x_conv = x_conv.transpose(1, 2)  # (B, L, d_inner)
        x_conv = F.silu(x_conv)

        # SSM
        ssm_state = state.get('ssm_state') if state else None
        y, new_ssm_state = self._ssm_sequential(x_conv, ssm_state=ssm_state)

        # Gate with z branch
        y = y * F.silu(z)

        # Output projection + residual
        output = self.out_proj(y) + residual

        new_state = {
            'conv_state': new_conv_state,
            'ssm_state': new_ssm_state.detach() if not self.training else new_ssm_state
        }

        return output, new_state


class MambaModel(nn.Module):
    """Stack of Mamba blocks for temporal sequence processing."""

    def __init__(self, d_model, n_layers, d_state=16, d_conv=4, expand=2):
        super().__init__()
        self.layers = nn.ModuleList([
            MambaBlock(d_model, d_state, d_conv, expand)
            for _ in range(n_layers)
        ])
        self.norm_f = RMSNorm(d_model)

    def forward(self, x, states=None):
        """
        Args:
            x: (B, L, d_model)
            states: list of per-layer state dicts, or None

        Returns:
            x: (B, L, d_model) - perception tokens
            new_states: list of per-layer state dicts
        """
        if states is None:
            states = [None] * len(self.layers)

        new_states = []
        for layer, state in zip(self.layers, states):
            x, new_state = layer(x, state)
            new_states.append(new_state)

        x = self.norm_f(x)
        return x, new_states