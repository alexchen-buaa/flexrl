from gym.envs.registration import register


register(
    id='KDTMaze-v0',
    entry_point='flexrl.envs.kdtmaze:KDTMaze',
    max_episode_steps=200
)

register(
    id='AKDTMaze-v0',
    entry_point='flexrl.envs.akdtmaze:AKDTMaze',
    max_episode_steps=200
)

register(
    id='KDTMazePixel-v0',
    entry_point='flexrl.envs.kdtmazepixel:KDTMazePixel',
    max_episode_steps=200
)

register(
    id='AKDTMazePixel-v0',
    entry_point='flexrl.envs.akdtmazepixel:AKDTMazePixel',
    max_episode_steps=200
)

register(
    id='KDTMazeFlat-v0',
    entry_point='flexrl.envs.kdtmazeflat:KDTMazeFlat',
    max_episode_steps=200
)

register(
    id='AKDTMazeFlat-v0',
    entry_point='flexrl.envs.akdtmazeflat:AKDTMazeFlat',
    max_episode_steps=200
)
