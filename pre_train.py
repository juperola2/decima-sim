from pretrain_compute_gradients import compute_actor_gradients
from spark_env.env import Environment, generate_coin_flips
from spark_env.timeline import Timeline
from pso_agent import PSOAgent
import numpy as np
from utils import *

from spark_env.wall_time import WallTime


def pre_train_actor_agent(actor_agent, seed, heuristic, num_eps, reset_prob, entropy_weight, lr):
    if heuristic != 'pso':
        return

    env = Environment()
    env.seed(seed)
    gradient_queue = []

    # global timer
    wall_time = WallTime()

    agent_pso = PSOAgent(wall_time)

    # set up storage for experience
    exp = {'node_inputs': [], 'job_inputs': [], \
           'gcn_mats': [], 'gcn_masks': [], \
           'summ_mats': [], 'running_dag_mat': [], \
           'dag_summ_back_mat': [], \
           'node_act_vec': [], 'job_act_vec': [], \
           'node_valid_mask': [], 'job_valid_mask': [], \
           'reward': [], 'wall_time': [],
           'job_state_change': [], 'node_act_probs': []}

    # ---- start pre-training process ----
    for ep in range(1, num_eps):
        print('pre-training epoch', ep)

        # generate max time stochastically based on reset prob
        max_time = generate_coin_flips(reset_prob)

        env.seed(seed)
        env.reset(max_time=max_time)

        gradient = train_with_pso(actor_agent, agent_pso, entropy_weight, env, exp)

        if gradient is not None:
            actor_agent.apply_gradients(gradient, lr)


def train_with_pso(actor_agent, agent_pso, entropy_weight, env, exp):
    # collect experiences
    while True:

        try:
            # The masking functions (node_valid_mask and
            # job_valid_mask in actor_agent.py) has some
            # small chance (once in every few thousand
            # iterations) to leave some non-zero probability
            # mass for a masked-out action. This will
            # trigger the check in "node_act and job_act
            # should be valid" in actor_agent.py
            # Whenever this is detected, we throw out the
            # rollout of that iteration and try again.

            # run experiment
            obs = env.observe()
            done = False
            pso_nodes_probs = []

            while not done:
                node, use_exec = invoke_model(actor_agent, obs, exp)
                pso_node, pso_limit = agent_pso.get_action(obs)
                pso_node_index = None

                job_dags, source_job, num_source_exec, \
                frontier_nodes, executor_limits, \
                exec_commit, moving_executors, action_map = obs
                for index, _node in action_map.map.items():
                    if _node == pso_node:
                        pso_node_index = index

                if node is None:
                    pso_node = node
                    pso_limit = use_exec

                pso_nodes_probs.append(pso_node_index)
                obs, reward, done = env.step(pso_node, pso_limit)

            actor_gradient, loss = compute_actor_gradients(
                actor_agent, exp, pso_nodes_probs, entropy_weight)
            return actor_gradient

        except AssertionError:
            print('Houve erro.')
            return None


def invoke_model(actor_agent, obs, exp):
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

    if sum(node_valid_mask[0, :]) == 0:
        # no node is valid to assign
        return None, num_source_exec

    # node_act should be valid
    assert node_valid_mask[0, node_act[0]] == 1

    # parse node action
    node = action_map[node_act[0]]

    # find job index based on node
    job_idx = job_dags.index(node.job_dag)

    # job_act should be valid
    assert job_valid_mask[0, job_act[0, job_idx] + \
        len(actor_agent.executor_levels) * job_idx] == 1

    # find out the executor limit decision
    if node.job_dag is source_job:
        agent_exec_act = actor_agent.executor_levels[
            job_act[0, job_idx]] - \
            exec_map[node.job_dag] + \
            num_source_exec
    else:
        agent_exec_act = actor_agent.executor_levels[
            job_act[0, job_idx]] - exec_map[node.job_dag]

    # parse job limit action
    use_exec = min(
        node.num_tasks - node.next_task_idx - \
        exec_commit.node_commit[node] - \
        moving_executors.count(node),
        agent_exec_act, num_source_exec)

    # for storing the action vector in experience
    node_act_vec = np.zeros(node_act_probs.shape)
    node_act_vec[0, node_act[0]] = 1

    # for storing job index
    job_act_vec = np.zeros(job_act_probs.shape)
    job_act_vec[0, job_idx, job_act[0, job_idx]] = 1

    # store experience
    exp['node_inputs'].append(node_inputs)
    exp['job_inputs'].append(job_inputs)
    exp['summ_mats'].append(summ_mats)
    exp['running_dag_mat'].append(running_dags_mat)
    exp['node_act_vec'].append(node_act_vec)
    exp['job_act_vec'].append(job_act_vec)
    exp['node_valid_mask'].append(node_valid_mask)
    exp['job_valid_mask'].append(job_valid_mask)
    exp['job_state_change'].append(job_dags_changed)
    exp['node_act_probs'].append(node_act_probs[0])

    if job_dags_changed:
        exp['gcn_mats'].append(gcn_mats)
        exp['gcn_masks'].append(gcn_masks)
        exp['dag_summ_back_mat'].append(dag_summ_backward_map)

    return node, use_exec
