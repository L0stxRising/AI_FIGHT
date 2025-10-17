import torch
import torch.nn as nn
import numpy as np
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
import pygame
STACK_N = 4
TARGET_H, TARGET_W = 144, 240
ACTION_LEN = 17
OUTINFO_LEN =  20
FightPlayer=None
class PlatformerTorchModelWithInfo(TorchModelV2, nn.Module):
    def __init__(self, obs_space, action_space, num_outputs, model_config, name):
        TorchModelV2.__init__(self, obs_space, action_space, num_outputs, model_config, name)
        nn.Module.__init__(self)
        self._conv = nn.Sequential(
            nn.Conv2d(STACK_N, 32, kernel_size=8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1), nn.ReLU(),
        )
        with torch.no_grad():
            dummy = torch.zeros(1, STACK_N, TARGET_H, TARGET_W)
            conv_out = self._conv(dummy)
            conv_out_size = int(np.prod(conv_out.shape[1:]))

        self._conv_out_size = conv_out_size
        self._outinfo_len = OUTINFO_LEN
        self._fc = nn.Sequential(
            nn.Linear(self._conv_out_size + self._outinfo_len, 512),
            nn.ReLU(),
        )

        self._policy_head = nn.Linear(512, int(np.sum(action_space.nvec)))
        self._value_head = nn.Linear(512, 1)
        self._features = None

    def forward(self, input_dict, state, seq_lens):
        """
        Expect input_dict["obs"] to be a dict:
          - input_dict["obs"]["obs"]: HxWxC uint8 (or float) image / batch of images
          - input_dict["obs"]["outInfo"]: 1D float vector (or batch) of length OUTINFO_LEN
        """
        obs_wrapper = input_dict["obs"]
        img = obs_wrapper["obs"]
        x = img.float() / 255.0 if isinstance(img, torch.Tensor) else torch.from_numpy(img).float() / 255.0
        if x.dim() == 3:
            x = x.unsqueeze(0)
        conv_out = self._conv(x).flatten(1)
        outInfo_raw = obs_wrapper.get("outInfo", None)
        if outInfo_raw is None:
            out_info = torch.zeros(conv_out.size(0), self._outinfo_len, device=conv_out.device, dtype=conv_out.dtype)
        else:
            out_info = outInfo_raw if isinstance(outInfo_raw, torch.Tensor) else torch.from_numpy(np.asarray(outInfo_raw)).float()
            if out_info.dim() == 1:
                out_info = out_info.unsqueeze(0).expand(conv_out.size(0), -1)
            elif out_info.size(0) != conv_out.size(0):
                if out_info.size(0) == 1:
                    out_info = out_info.expand(conv_out.size(0), -1)
                else:
                    out_info = out_info.reshape(out_info.size(0), -1)
                    out_info = out_info[:conv_out.size(0), :self._outinfo_len]
                    if out_info.size(1) < self._outinfo_len:
                        pad = torch.zeros(out_info.size(0), self._outinfo_len - out_info.size(1), device=conv_out.device)
                        out_info = torch.cat([out_info, pad], dim=1)
            out_info = out_info.to(conv_out.device, dtype=conv_out.dtype)

        combined = torch.cat([conv_out, out_info], dim=1)

        self._features = self._fc(combined)
        logits = self._policy_head(self._features)
        return logits, state

    def value_function(self):
        assert self._features is not None, "call forward first"
        return self._value_head(self._features).squeeze(1)
# -------------------- Imports --------------------
import sys
import numpy as np
import torch
from gymnasium.spaces import Box, MultiDiscrete, Dict
from ray.rllib.env.multi_agent_env import MultiAgentEnv
import Game_Module as game_module
# -------------------- Human Controls --------------------
KEY_BINDINGS = {
    "jump": pygame.K_w,
    "left": pygame.K_a,
    "right": pygame.K_d,
    "fire": pygame.K_j,
    "saw": pygame.K_k,
    "ice": pygame.K_l,
    "slash": pygame.K_u,
    "kick": pygame.K_i,
    "Deathbeam": pygame.K_o,
    "spikeBall": pygame.K_p,
    "BlackHole": pygame.K_h,
    "SpikeGnd": pygame.K_n,
    "dash": pygame.K_SPACE,
    "Invis": pygame.K_q,
    "Parry": pygame.K_e,
    "doubleJump": pygame.K_s,
}
def get_player_actions():
    keys = pygame.key.get_pressed()
    actions = [0] * len(game_module.Controls)

    for idx, action in enumerate(game_module.Controls):
        if action == "Noop":
            continue
        if action in KEY_BINDINGS and keys[KEY_BINDINGS[action]]:
            actions[idx] = 1

    return actions
# -------------------- Constants --------------------
STACK_N = 4
TARGET_H = 144
TARGET_W = 240
ACTION_LEN = 17
OUTINFO_DIM = 20
MAX_STEPS = 60 * 60
# -------------------- Spaces --------------------
ACT_SPACE = MultiDiscrete([2] * ACTION_LEN)
OBS_SPACE = Dict({
    "obs": Box(low=0, high=255, shape=(TARGET_H, TARGET_W, STACK_N), dtype=np.uint8),
    "outInfo": Box(low=-np.inf, high=np.inf, shape=(OUTINFO_DIM,), dtype=np.float32)
})
env=game_module.Env()
NUM_OUTPUTS = int(np.sum(ACT_SPACE.nvec))
DUMMY_CONFIG = {}
import os
file=os.path.abspath(__file__).split("\\")
file=file[0:len(file)-1]
fpath="\\".join(file)
modelP_1 = PlatformerTorchModelWithInfo(OBS_SPACE, ACT_SPACE, NUM_OUTPUTS, DUMMY_CONFIG, "player1")
modelP_2 = PlatformerTorchModelWithInfo(OBS_SPACE, ACT_SPACE, NUM_OUTPUTS, DUMMY_CONFIG, "player2")
stateDict_1=torch.load(os.path.join(fpath,r"Policies\player_1_model-600 (2).pth"),map_location=torch.device('cpu'))
stateDict_2=torch.load(os.path.join(fpath,r"Policies\player_2_model-600 (2).pth"),map_location=torch.device('cpu'))
modelP_1.load_state_dict(stateDict_1)
modelP_2.load_state_dict(stateDict_2)
modelP_1.eval()
modelP_2.eval()
frame=game_module.get_state(game_module.WIN).unsqueeze(0)
_,outInfo=game_module.getState(env.game.player1,env.game.player2)
clock = pygame.time.Clock()
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
    clock.tick(60)
    obs_tensor = torch.tensor(frame, dtype=torch.float32)
    outInfo_tensor1 = torch.tensor(outInfo[0], dtype=torch.float32).unsqueeze(0)
    outInfo_tensor2 = torch.tensor(outInfo[1], dtype=torch.float32).unsqueeze(0)
    if FightPlayer == 1:
        outInfo_tensor1 = torch.tensor(outInfo[0], dtype=torch.float32).unsqueeze(0)
        pred1, _ = modelP_1({"obs": {"obs": obs_tensor, "outInfo": outInfo_tensor1}}, [], None)
        probs1= torch.sigmoid(pred1)
        dist1= torch.distributions.Bernoulli(probs1)
        actions1 = dist1.sample().squeeze(0).tolist()
        actions2=get_player_actions()
    if FightPlayer == 2:
        outInfo_tensor2 = torch.tensor(outInfo[1], dtype=torch.float32).unsqueeze(0)
        pred2, _ = modelP_2({"obs": {"obs": obs_tensor, "outInfo": outInfo_tensor2}}, [], None)
        probs2= torch.sigmoid(pred2)
        dist2= torch.distributions.Bernoulli(probs2)
        actions2 = dist2.sample().squeeze(0).tolist()
        actions1=get_player_actions()
    if FightPlayer==None:
        outInfo_tensor1 = torch.tensor(outInfo[0], dtype=torch.float32).unsqueeze(0) 
        outInfo_tensor2 = torch.tensor(outInfo[1], dtype=torch.float32).unsqueeze(0)
        pred1, _ = modelP_1({"obs": {"obs": obs_tensor, "outInfo": outInfo_tensor1}}, [], None)
        probs1= torch.sigmoid(pred1)
        dist1= torch.distributions.Bernoulli(probs1)
        actions1 = dist1.sample().squeeze(0).tolist()
        pred2, _ = modelP_2({"obs": {"obs": obs_tensor, "outInfo": outInfo_tensor2}}, [], None)
        probs2= torch.sigmoid(pred2)
        dist2= torch.distributions.Bernoulli(probs2)
        actions2 = dist2.sample().squeeze(0).tolist()
    print("player_1 : ",actions1)
    print("player_2 : ",actions2)
    Newframe,outInfo,reward1,reward2,done=env.step(actions2, actions1,True)
    if done:
        print(env.P_1)
        print(env.P_2)
        env.reset()
    frame=Newframe.unsqueeze(0)