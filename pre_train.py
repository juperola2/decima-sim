from spark_env.env import Environment, generate_coin_flips
from spark_env.job_generator import generate_jobs
from spark_env.timeline import Timeline

import numpy as np

from spark_env.wall_time import WallTime


def pre_train_actor_agent(actor_agent, seed, heuristic, num_eps, reset_prob):
    if heuristic != 'pso_init':
        return

    env = Environment()
    # initialize episode reset probability
    env.seed(seed)

    timeline = Timeline()

    # global timer
    wall_time = WallTime()

    # generate max time stochastically based on reset prob
    max_time = generate_coin_flips(reset_prob)

    env.reset(max_time=max_time)

    # ---- start pre-training process ----
    for ep in range(1, num_eps):
        print('pre-training epoch', ep)

        np_random = np.random.RandomState()

        # generate a set of new jobs
        job_dags = generate_jobs(
            np_random, timeline, wall_time)

        # generate max time stochastically based on reset prob
        max_time = generate_coin_flips(reset_prob)

        # uses priority queue


        node, use_exec = invoke_model(actor_agent)


def invoke_model(actor_agent, obs):
    # parse observation
    job_dags, source_job, num_source_exec, \
    frontier_nodes, executor_limits, \
    exec_commit, moving_executors, action_map = obs

    if len(frontier_nodes) == 0:
        # no action to take
        return None, num_source_exec

    # invoking the learning model
    node_act, job_act, \
        node_act_probs, job_act_probs, \
        node_inputs, job_inputs, \
        node_valid_mask, job_valid_mask, \
        gcn_mats, gcn_masks, summ_mats, \
        running_dags_mat, dag_summ_backward_map, \
        exec_map, job_dags_changed = \
            actor_agent.invoke_model(obs)


