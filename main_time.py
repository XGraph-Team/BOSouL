# The scalability of the methods and the baselines
# evaluated on different sizes of SW graphs

import warnings
import gc

warnings.filterwarnings("ignore")

from methods import *
import time
import statistics as s

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

################################################
# Global parameters
################################################
diffusion_model = "si"
infect_rate = 0.1
graph_size = 1000
candidate_size = 50
seed_size = 3
actual_time_step_size = 15
num_iterations = 30
num_of_sims = 50

for n in [1000,2000,3000,4000,5000]:
    gc.collect()

    print('===============================================================')
    print('n = ', n)

    BOSouLT = []
    JORDANT = []
    LISNT = []
    NETT = []

    for i in range(5):

        # generate a SW graph
        G = connSW(n)

        ################################################
        # Set up ground truth
        ################################################
        s_star = create_true_source_set(G, num_of_sources=seed_size)
        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=s_star)
        contagion.forward(actual_time_step_size)

        obs = contagion.get_infected_subgraph(step=actual_time_step_size - 1)
        c_star = list(obs.nodes)

        ################################################
        # peak_mean for ground truth source set
        ################################################
        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=s_star)
        peak_mean, peak_var = source_coverage(contagion, c_star, num_of_sims)

        ################################################
        # methods evaluation
        ################################################
        start = time.time()
        gpsi_cs_knn = GPSI_cluster_sampling(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size,
                                            diffusion_model, seed_size, number_of_clusters=20)
        end = time.time()
        BOSouLT.append(end - start)

        start = time.time()
        jordan = cosasi.source_inference.multiple_source.fast_multisource_jordan_centrality(obs, G, 3).topn(1)
        end = time.time()
        JORDANT.append(end - start)

        start = time.time()
        lisn = cosasi.source_inference.multiple_source.fast_multisource_lisn(obs, G, actual_time_step_size, 3).topn((1))
        end = time.time()
        LISNT.append(end - start)

        start = time.time()
        sleuth = cosasi.source_inference.multiple_source.fast_multisource_netsleuth(obs, G, number_sources=3).topn((1))
        end = time.time()
        NETT.append(end - start)

        

    print('BOSouL: ', s.mean(BOSouLT), '+-', s.stdev(BOSouLT))
    print('JORDAN: ', s.mean(JORDANT), '+-', s.stdev(JORDANT))
    print('LISN: ', s.mean(LISNT), '+-', s.stdev(LISNT))
    print('NET: ', s.mean(NETT), '+-', s.stdev(NETT))

    gc.collect()
