"""FiLM-conditioned Action Expert used by the original Route 2 baseline."""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Beta


class SinusoidalTimeEmbedding(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, timestep):
        half = self.dim // 2
        exponent = -torch.arange(half, dtype=torch.float32, device=timestep.device) * (
            math.log(10000.0) / half
        )
        frequencies = timestep.float().unsqueeze(-1) * exponent.exp()
        return torch.cat((torch.sin(frequencies), torch.cos(frequencies)), dim=-1).to(
            timestep.dtype
        )


class AdaRMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.eps = eps
        self.modulation = nn.Linear(dim, dim * 3)
        nn.init.zeros_(self.modulation.weight)
        nn.init.zeros_(self.modulation.bias)

    def forward(self, x, condition):
        variance = x.float().pow(2).mean(-1, keepdim=True)
        normalized = (x * torch.rsqrt(variance + self.eps)).to(x.dtype)
        scale, shift, gate = self.modulation(condition).chunk(3, dim=-1)
        normalized = normalized * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
        return normalized, gate.unsqueeze(1)


class SwiGLUFeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim)
        self.up_proj = nn.Linear(dim, hidden_dim)
        self.down_proj = nn.Linear(hidden_dim, dim)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class ActionFiLM(nn.Module):
    def __init__(self, lang_dim, hidden_dim, num_layers, film_hidden=None):
        super().__init__()
        film_hidden = film_hidden or hidden_dim
        self.norm = nn.LayerNorm(lang_dim)
        self.trunk = nn.Sequential(nn.Linear(lang_dim, film_hidden), nn.GELU())
        self.heads = nn.ModuleList(
            [nn.Linear(film_hidden, 2 * hidden_dim) for _ in range(num_layers)]
        )
        for head in self.heads:
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def forward(self, language, language_mask=None):
        if language_mask is None:
            pooled = language.mean(dim=1)
        else:
            weight = language_mask.unsqueeze(-1).to(language.dtype)
            pooled = (language * weight).sum(dim=1) / weight.sum(dim=1).clamp_min(1e-6)
        hidden = self.trunk(self.norm(pooled))
        return [head(hidden).chunk(2, dim=-1) for head in self.heads]


class ActionCrossSelfAttention(nn.Module):
    """Action queries attend to Predictor K/V and the action chunk's own K/V."""

    def __init__(self, hidden_size, num_heads, head_dim):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = head_dim
        inner_dim = num_heads * head_dim
        self.q_proj = nn.Linear(hidden_size, inner_dim)
        self.k_proj = nn.Linear(hidden_size, inner_dim)
        self.v_proj = nn.Linear(hidden_size, inner_dim)
        self.o_proj = nn.Linear(inner_dim, hidden_size)

    def forward(self, action_hidden, context_key, context_value):
        batch_size, action_length, _ = action_hidden.shape
        query = (
            self.q_proj(action_hidden)
            .view(batch_size, action_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        action_key = (
            self.k_proj(action_hidden)
            .view(batch_size, action_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        action_value = (
            self.v_proj(action_hidden)
            .view(batch_size, action_length, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        key = torch.cat((context_key, action_key), dim=2)
        value = torch.cat((context_value, action_value), dim=2)
        output = F.scaled_dot_product_attention(query, key, value)
        output = output.transpose(1, 2).contiguous().view(batch_size, action_length, -1)
        return self.o_proj(output)


class ActionBlock(nn.Module):
    def __init__(self, hidden_size, num_heads, head_dim, ffn_mult=4):
        super().__init__()
        self.pre_attn_norm = AdaRMSNorm(hidden_size)
        self.attn = ActionCrossSelfAttention(hidden_size, num_heads, head_dim)
        self.pre_ffn_norm = AdaRMSNorm(hidden_size)
        self.ffn = SwiGLUFeedForward(hidden_size, hidden_size * ffn_mult)

    def forward(self, x, context_key, context_value, time_condition, film):
        if film is not None:
            gamma, beta = film
            x = x * (1.0 + gamma.unsqueeze(1)) + beta.unsqueeze(1)
        normalized, attention_gate = self.pre_attn_norm(x, time_condition)
        x = x + self.attn(normalized, context_key, context_value) * attention_gate
        normalized, ffn_gate = self.pre_ffn_norm(x, time_condition)
        return x + self.ffn(normalized) * ffn_gate


class ActionExpert(nn.Module):
    """Historical DiT Action Expert with explicit language FiLM conditioning."""

    def __init__(
        self,
        action_dim,
        action_chunk_size,
        num_layers,
        num_heads,
        head_dim,
        hidden_size=384,
        ffn_mult=4,
        lang_dim=4096,
        film_hidden=None,
        num_inference_steps=10,
        noise_beta_alpha=1.5,
        noise_beta_beta=1.0,
        num_timestep_buckets=1000,
        read_full_predictor_kv=False,
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_chunk_size = action_chunk_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.head_dim = head_dim
        self.num_inference_steps = num_inference_steps
        self.num_timestep_buckets = num_timestep_buckets
        self.read_full_predictor_kv = read_full_predictor_kv
        self._beta_alpha = noise_beta_alpha
        self._beta_beta = noise_beta_beta

        self.action_in_proj = nn.Linear(action_dim * 2, hidden_size)
        self.time_sinusoidal = SinusoidalTimeEmbedding(hidden_size)
        self.time_mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
        )
        self.position_embedding = nn.Embedding(action_chunk_size, hidden_size)
        nn.init.normal_(self.position_embedding.weight, std=0.02)
        self.film = ActionFiLM(lang_dim, hidden_size, num_layers, film_hidden)
        self.blocks = nn.ModuleList(
            [
                ActionBlock(hidden_size, num_heads, head_dim, ffn_mult)
                for _ in range(num_layers)
            ]
        )
        self.final_norm = AdaRMSNorm(hidden_size)
        self.out_proj = nn.Linear(hidden_size, action_dim)

    def sample_time(self, batch_size, device, dtype):
        distribution = Beta(self._beta_alpha, self._beta_beta)
        sample = (
            distribution.sample([batch_size]).to(device, dtype=dtype).clamp(max=0.999)
        )
        return (0.999 - sample) / 0.999

    def _time_cond(self, timestep):
        return self.time_mlp(self.time_sinusoidal(timestep.float()))

    def _encode_actions(self, actions, dof_mask):
        dof_mask = dof_mask if dof_mask is not None else torch.ones_like(actions)
        tokens = self.action_in_proj(
            torch.cat((actions, dof_mask.to(actions.dtype)), dim=-1)
        )
        positions = torch.arange(tokens.shape[1], device=actions.device)
        return tokens + self.position_embedding(positions).unsqueeze(0)

    def _run_blocks(self, tokens, predictor_kv, context_len, time_condition, film):
        if len(predictor_kv) != self.num_layers:
            raise ValueError(
                f"ActionExpert has {self.num_layers} blocks but predictor supplied "
                f"{len(predictor_kv)} layers"
            )
        hidden = tokens
        for index, block in enumerate(self.blocks):
            context_key, context_value = predictor_kv[index]
            if context_len is not None and not self.read_full_predictor_kv:
                context_key = context_key[:, :, :context_len]
                context_value = context_value[:, :, :context_len]
            hidden = block(
                hidden,
                context_key,
                context_value,
                time_condition,
                film[index] if film is not None else None,
            )
        hidden, _ = self.final_norm(hidden, time_condition)
        return self.out_proj(hidden)

    def forward(
        self,
        predictor_kv,
        actions,
        lang,
        lang_mask=None,
        context_len=None,
        dof_mask=None,
    ):
        noise = torch.randn_like(actions)
        time = self.sample_time(actions.shape[0], actions.device, actions.dtype)
        noisy = (1 - time[:, None, None]) * noise + time[:, None, None] * actions
        target_velocity = actions - noise
        timestep = (time * self.num_timestep_buckets).long()
        predicted_velocity = self._run_blocks(
            self._encode_actions(noisy, dof_mask),
            predictor_kv,
            context_len,
            self._time_cond(timestep),
            self.film(lang, lang_mask),
        )
        return predicted_velocity, target_velocity

    @torch.no_grad()
    def predict_action(
        self,
        predictor_kv,
        lang,
        lang_mask=None,
        context_len=None,
        dof_mask=None,
    ):
        first_key = predictor_kv[0][0]
        actions = torch.randn(
            (first_key.shape[0], self.action_chunk_size, self.action_dim),
            device=first_key.device,
            dtype=first_key.dtype,
        )
        film = self.film(lang, lang_mask)
        step_size = 1.0 / self.num_inference_steps
        for step in range(self.num_inference_steps):
            timestep = torch.full(
                (actions.shape[0],),
                int(step / self.num_inference_steps * self.num_timestep_buckets),
                device=actions.device,
                dtype=torch.long,
            )
            velocity = self._run_blocks(
                self._encode_actions(actions, dof_mask),
                predictor_kv,
                context_len,
                self._time_cond(timestep),
                film,
            )
            actions = actions + step_size * velocity
        return actions
