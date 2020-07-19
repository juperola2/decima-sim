from agent import Agent

import numpy as np


class PSOAgent(Agent):
    def __init__(self, wall_time, swarm_size=100, num_it=50, c1=0.5, c2=0.8, w_max=1.4, w_min=0.6):
        self.particles = []
        self.gbest = None
        self.c1 = c1
        self.c2 = c2
        self.w = w_max
        self.w_max = w_max
        self.w_min = w_min
        self.num_it = num_it  # T
        self.swarm_size = swarm_size
        self.d = 0
        self.env_wall_time = wall_time

    def compute_limits(self):
        i, c = np.unique(self.gbest.position, return_counts=True)
        m = c.argmax()
        return i[m], c[m]

    def get_action(self, obs):

        # parse observation
        job_dags, source_job, num_source_exec, \
        frontier_nodes, executor_limits, \
        exec_commit, moving_executors, action_map = obs
        self.nodes = list(frontier_nodes)
        n = len(frontier_nodes)
        if n == 0: #confirmar, mas a principio não teria mais jobs para escalonar
            return None, num_source_exec

        self.pso(num_source_exec, n)

        #TODO - definir a estratégia de fornecer 1 nó (no caso do teste)
        n, l = self.compute_limits()
        limit = min(l, num_source_exec)
        node = self.nodes[n]

        return node, limit

    def pso(self,  num_source_exec, nodes):
        # faz um escalonamento.

        # prepara o agente pro evento de escalonamento

        self.reset_agent(num_source_exec, nodes)
        self.init_particles()

        for i in range(self.num_it):
            self.update_w(i)
            # Update best
            for p in self.particles:
                if p.fitness > p.best.fitness:
                    p.update_pbest()
                if p.fitness > self.gbest.fitness:
                    self.gbest = p.best
            # Update particles
            for p in self.particles:
                phi = np.random.random(2)
                # update velocity
                p.velocity = self.w * p.velocity \
                             + self.c1 * phi[0] \
                             *(p.best.position - p.position) \
                             + self.c2 * phi[1] \
                             * (self.gbest.position - p.position)
                # update position
                p.update_position()
                # update fitness fitness
                p.compute_fitness()

        return self.gbest

    def update_w(self, t):
        # TODO - rever essa atualização do W (eq 10).
        if t < (0.7 * self.num_it):
            self.w = self.w_max - t * (self.w_max - self.w_min) / self.num_it
        else:
            self.w = 0.4 + 0.3 * np.random.random(1).item()


    def init_particles(self):
        rng = np.random.default_rng()
        for i in range(self.swarm_size):
            p = Particle(self, rng.integers(0, self.swarm_size, self.d), rng.integers(0, self.swarm_size, self.d), self.d)
            p.compute_fitness()
            p.best = Best(p.position, p.fitness)
            self.particles.append(p)
        self.gbest = self.particles[0].best

    def reset_agent(self, num_free_executors, num_nodes):  # prepara os atributos pro evento de escal. atual...

        self.d = num_free_executors
        self.swarm_size = num_nodes
        self.w = self.w_max
        self.particles = []
        self.gbest = Best()


class Particle(object):

    def __init__(self, pso, p=None, v=None, d=None):
        self.best = None
        self.velocity = v
        self.position = p
        self.fitness = None
        self.d = d
        self.pso = pso

    def compute_fitness(self):
        # TODO - definir a função de fitness
        # TODO - pegar a função usada pra JCT e pegar o fitness do evento passado (?)
        f = 0.0
        for n in self.pso.nodes:
            f = f + self.pso.env_wall_time.curr_time *((n.job_dag.num_nodes - n.job_dag.num_nodes_done)/n.job_dag.num_nodes)
        self.fitness = -f

        pass

    def update_pbest(self):
        self.best.position = np.copy(self.position)
        self.best.fitness = self.fitness

    def update_position(self):
        # TODO - revisar essa conversão para inteiro
        self.position = np.floor(np.mod((self.position + self.velocity), np.array([self.d])))


class Best(object):
    def __init__(self, p=None, f=None):
        self.position = p
        self.fitness = f
