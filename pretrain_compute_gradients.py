from param import *
from utils import *
from sparse_op import expand_sp_mat, merge_and_extend_sp_mat


def compute_actor_gradients(actor_agent, exp, pso_nodes_probs, entropy_weight):
    batch_points = truncate_experiences(exp['job_state_change'])

    all_gradients = []
    all_loss = [[], [], 0]

    for b in range(len(batch_points) - 1):
        # need to do different batches because the
        # size of dags in state changes
        ba_start = batch_points[b]
        ba_end = batch_points[b + 1]

        # use a piece of experience
        node_inputs = np.vstack(exp['node_inputs'][ba_start : ba_end])
        job_inputs = np.vstack(exp['job_inputs'][ba_start : ba_end])
        node_act_vec = np.vstack(exp['node_act_vec'][ba_start : ba_end])
        job_act_vec = np.vstack(exp['job_act_vec'][ba_start : ba_end])
        node_valid_mask = np.vstack(exp['node_valid_mask'][ba_start : ba_end])
        job_valid_mask = np.vstack(exp['job_valid_mask'][ba_start : ba_end])
        summ_mats = exp['summ_mats'][ba_start : ba_end]
        running_dag_mats = exp['running_dag_mat'][ba_start : ba_end]
        node_act_probs = exp['node_act_probs'][ba_start: ba_end]

        adv = np.zeros((node_act_vec.shape[0], 1))
        pso_node_indexes = pso_nodes_probs[ba_start: ba_end]
        i = 0
        for pso_node_index, node_act_prob in zip (pso_node_indexes, node_act_probs):
            probs = [0.0] * node_act_vec.shape[1]
            # TODO:: verificar pq as vezes o indice é maior ou igual ao tamanho vetor de probabilidades
            if pso_node_index is not None and pso_node_index < node_act_vec.shape[1]:
                probs[pso_node_index] = 1.0

            probs = np.array(probs)
            adv[i] = np.sum((node_act_prob - probs) ** 2)
            i += 1

        gcn_mats = exp['gcn_mats'][b]
        gcn_masks = exp['gcn_masks'][b]
        summ_backward_map = exp['dag_summ_back_mat'][b]

        # given an episode of experience (advantage computed from baseline)
        batch_size = node_act_vec.shape[0]

        # expand sparse adj_mats
        extended_gcn_mats = expand_sp_mat(gcn_mats, batch_size)

        # extended masks
        # (on the dimension according to extended adj_mat)
        extended_gcn_masks = [np.tile(m, (batch_size, 1)) for m in gcn_masks]

        # expand sparse summ_mats
        extended_summ_mats = merge_and_extend_sp_mat(summ_mats)

        # expand sparse running_dag_mats
        extended_running_dag_mats = merge_and_extend_sp_mat(running_dag_mats)

        # compute gradient
        act_gradients, loss = actor_agent.get_gradients(
            node_inputs, job_inputs,
            node_valid_mask, job_valid_mask,
            extended_gcn_mats, extended_gcn_masks,
            extended_summ_mats, extended_running_dag_mats,
            summ_backward_map, node_act_vec, job_act_vec,
            adv, entropy_weight)

        all_gradients.append(act_gradients)
        all_loss[0].append(loss[0])
        all_loss[1].append(loss[1])

    all_loss[0] = np.sum(all_loss[0])
    all_loss[1] = np.sum(all_loss[1])  # to get entropy

    # aggregate all gradients from the batches
    gradients = aggregate_gradients(all_gradients)

    return gradients, all_loss
