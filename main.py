# The performance of the methods and the baselines given the number of sources
# on multiple graphs: Planetoid, ER, and SW

import warnings

warnings.filterwarnings("ignore")

from methods import *
import time
import statistics as s

print(time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()))

################################################
# Global parameters
################################################
num_of_sims = 50

print('small world')
G = connSW(1000)

GPCS = []
CST = []
GPRS = []
RST = []
GPSI = []
SIT = []
JORDAN = []
JORDANT = []
LISN = []
LISNT = []
NET = []
NETT = []

for i in range(10):

    s_star = create_true_source_set(G, num_of_sources=seed_size)
    contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=s_star)
    contagion.forward(actual_time_step_size)
    obs = contagion.get_infected_subgraph(step=actual_time_step_size-1)
    c_star = list(obs.nodes)
    
    start = time.time()
    gpsi_cs_kmeans = GPSI_cluster_sampling(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size, number_of_clusters=20)
    end = time.time()
    CST.append(end - start)
    dist = node_set_distance(gpsi_cs_kmeans, s_star, G)
    GPCS.append(dist)

    start = time.time()
    gpsi_ft = GPSI_ft(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size)
    end = time.time()
    RST.append(end - start)
    dist = node_set_distance(gpsi_ft, s_star, G)
    GPRS.append(dist)

    start = time.time()
    gpsi_vanilla = GPSI_vanilla(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size)
    end = time.time()
    SIT.append(end - start)
    dist = node_set_distance(gpsi_vanilla, s_star, G)
    GPSI.append(dist)

    start = time.time()
    jordan = cosasi.source_inference.multiple_source.fast_multisource_jordan_centrality(obs, G, 3).topn(1)
    end = time.time()
    jordan_result = []
    for j in jordan[0]:
        jordan_result.append(j)
    dist = node_set_distance(jordan_result, s_star, G)
    JORDANT.append(end - start)
    JORDAN.append(dist)

    start = time.time()
    lisn = cosasi.source_inference.multiple_source.fast_multisource_lisn(obs, G, actual_time_step_size, 3).topn((1))
    end = time.time()
    lisn_result = []
    for j in lisn[0]:
        lisn_result.append(j)
    dist = node_set_distance(lisn_result, s_star, G)
    LISNT.append(end - start)
    LISN.append(dist)

    start = time.time()
    sleuth = cosasi.source_inference.multiple_source.fast_multisource_netsleuth(obs, G, number_sources=3).topn((1))
    end = time.time()
    sleuth_result = []
    for j in sleuth[0]:
        sleuth_result.append(j)
    dist = node_set_distance(sleuth_result, s_star, G)
    NETT.append(end - start)
    NET.append(dist)

print('GPSI with fourier transfer and kmeans cluster sampling')
print('time: ', s.mean(CST), "+-", s.stdev(CST))
print('eval: ', s.mean(GPCS), '+-', s.stdev(GPCS))

print('GPSI with fourier transfer and random sampling')
print('time: ', s.mean(RST), "+-", s.stdev(RST))
print('eval: ', s.mean(GPRS), '+-', s.stdev(GPRS))

print('GPSI with vanilla sampling')
print('time: ', s.mean(SIT), "+-", s.stdev(SIT))
print('eval: ', s.mean(GPSI), '+-', s.stdev(GPSI))

print('Jordan sampling')
print('time: ', s.mean(JORDANT), "+-", s.stdev(JORDANT))
print('eval: ', s.mean(JORDAN), '+-', s.stdev(JORDAN))

print('LISN sampling')
print('time: ', s.mean(LISNT), "+-", s.stdev(LISNT))
print('eval: ', s.mean(LISN), '+-', s.stdev(LISN))

print('NET sampling')
print('time: ', s.mean(NETT), "+-", s.stdev(NETT))
print('eval: ', s.mean(NET), '+-', s.stdev(NET))


print('ER')
G = ER(1000)

GPCS = []
CST = []
GPRS = []
RST = []
GPSI = []
SIT = []
JORDAN = []
JORDANT = []
LISN = []
LISNT = []
NET = []
NETT = []

for i in range(10):

    s_star = create_true_source_set(G, num_of_sources=seed_size)
    contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=s_star)
    contagion.forward(actual_time_step_size)
    obs = contagion.get_infected_subgraph(step=actual_time_step_size-1)
    c_star = list(obs.nodes)
    
    start = time.time()
    gpsi_cs_kmeans = GPSI_cluster_sampling(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size, number_of_clusters=20)
    end = time.time()
    CST.append(end - start)
    dist = node_set_distance(gpsi_cs_kmeans, s_star, G)
    GPCS.append(dist)

    start = time.time()
    gpsi_ft = GPSI_ft(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size)
    end = time.time()
    RST.append(end - start)
    dist = node_set_distance(gpsi_ft, s_star, G)
    GPRS.append(dist)

    start = time.time()
    gpsi_vanilla = GPSI_vanilla(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size)
    end = time.time()
    SIT.append(end - start)
    dist = node_set_distance(gpsi_vanilla, s_star, G)
    GPSI.append(dist)

    start = time.time()
    jordan = cosasi.source_inference.multiple_source.fast_multisource_jordan_centrality(obs, G, 3).topn(1)
    end = time.time()
    jordan_result = []
    for j in jordan[0]:
        jordan_result.append(j)
    dist = node_set_distance(jordan_result, s_star, G)
    JORDANT.append(end - start)
    JORDAN.append(dist)

    start = time.time()
    lisn = cosasi.source_inference.multiple_source.fast_multisource_lisn(obs, G, actual_time_step_size, 3).topn((1))
    end = time.time()
    lisn_result = []
    for j in lisn[0]:
        lisn_result.append(j)
    dist = node_set_distance(lisn_result, s_star, G)
    LISNT.append(end - start)
    LISN.append(dist)

    start = time.time()
    sleuth = cosasi.source_inference.multiple_source.fast_multisource_netsleuth(obs, G, number_sources=3).topn((1))
    end = time.time()
    sleuth_result = []
    for j in sleuth[0]:
        sleuth_result.append(j)
    dist = node_set_distance(sleuth_result, s_star, G)
    NETT.append(end - start)
    NET.append(dist)

print('GPSI with fourier transfer and kmeans cluster sampling')
print('time: ', s.mean(CST), "+-", s.stdev(CST))
print('eval: ', s.mean(GPCS), '+-', s.stdev(GPCS))

print('GPSI with fourier transfer and random sampling')
print('time: ', s.mean(RST), "+-", s.stdev(RST))
print('eval: ', s.mean(GPRS), '+-', s.stdev(GPRS))

print('GPSI with vanilla sampling')
print('time: ', s.mean(SIT), "+-", s.stdev(SIT))
print('eval: ', s.mean(GPSI), '+-', s.stdev(GPSI))

print('Jordan sampling')
print('time: ', s.mean(JORDANT), "+-", s.stdev(JORDANT))
print('eval: ', s.mean(JORDAN), '+-', s.stdev(JORDAN))

print('LISN sampling')
print('time: ', s.mean(LISNT), "+-", s.stdev(LISNT))
print('eval: ', s.mean(LISN), '+-', s.stdev(LISN))

print('NET sampling')
print('time: ', s.mean(NETT), "+-", s.stdev(NETT))
print('eval: ', s.mean(NET), '+-', s.stdev(NET))

print('Cora')
G = Cora()

GPCS = []
CST = []
GPRS = []
RST = []
GPSI = []
SIT = []
JORDAN = []
JORDANT = []
LISN = []
LISNT = []
NET = []
NETT = []

for i in range(10):

    s_star = create_true_source_set(G, num_of_sources=seed_size)
    contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=s_star)
    contagion.forward(actual_time_step_size)
    obs = contagion.get_infected_subgraph(step=actual_time_step_size-1)
    c_star = list(obs.nodes)
    
    start = time.time()
    gpsi_cs_kmeans = GPSI_cluster_sampling(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size, number_of_clusters=20)
    end = time.time()
    CST.append(end - start)
    dist = node_set_distance(gpsi_cs_kmeans, s_star, G)
    GPCS.append(dist)

    start = time.time()
    gpsi_ft = GPSI_ft(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size)
    end = time.time()
    RST.append(end - start)
    dist = node_set_distance(gpsi_ft, s_star, G)
    GPRS.append(dist)

    start = time.time()
    gpsi_vanilla = GPSI_vanilla(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size)
    end = time.time()
    SIT.append(end - start)
    dist = node_set_distance(gpsi_vanilla, s_star, G)
    GPSI.append(dist)

    start = time.time()
    jordan = cosasi.source_inference.multiple_source.fast_multisource_jordan_centrality(obs, G, 3).topn(1)
    end = time.time()
    jordan_result = []
    for j in jordan[0]:
        jordan_result.append(j)
    dist = node_set_distance(jordan_result, s_star, G)
    JORDANT.append(end - start)
    JORDAN.append(dist)

    start = time.time()
    lisn = cosasi.source_inference.multiple_source.fast_multisource_lisn(obs, G, actual_time_step_size, 3).topn((1))
    end = time.time()
    lisn_result = []
    for j in lisn[0]:
        lisn_result.append(j)
    dist = node_set_distance(lisn_result, s_star, G)
    LISNT.append(end - start)
    LISN.append(dist)

    start = time.time()
    sleuth = cosasi.source_inference.multiple_source.fast_multisource_netsleuth(obs, G, number_sources=3).topn((1))
    end = time.time()
    sleuth_result = []
    for j in sleuth[0]:
        sleuth_result.append(j)
    dist = node_set_distance(sleuth_result, s_star, G)
    NETT.append(end - start)
    NET.append(dist)

print('GPSI with fourier transfer and kmeans cluster sampling')
print('time: ', s.mean(CST), "+-", s.stdev(CST))
print('eval: ', s.mean(GPCS), '+-', s.stdev(GPCS))

print('GPSI with fourier transfer and random sampling')
print('time: ', s.mean(RST), "+-", s.stdev(RST))
print('eval: ', s.mean(GPRS), '+-', s.stdev(GPRS))

print('GPSI with vanilla sampling')
print('time: ', s.mean(SIT), "+-", s.stdev(SIT))
print('eval: ', s.mean(GPSI), '+-', s.stdev(GPSI))

print('Jordan sampling')
print('time: ', s.mean(JORDANT), "+-", s.stdev(JORDANT))
print('eval: ', s.mean(JORDAN), '+-', s.stdev(JORDAN))

print('LISN sampling')
print('time: ', s.mean(LISNT), "+-", s.stdev(LISNT))
print('eval: ', s.mean(LISN), '+-', s.stdev(LISN))

print('NET sampling')
print('time: ', s.mean(NETT), "+-", s.stdev(NETT))
print('eval: ', s.mean(NET), '+-', s.stdev(NET))


print('CiteSeer')
G = CiteSeer()

GPCS = []
CST = []
GPRS = []
RST = []
GPSI = []
SIT = []
JORDAN = []
JORDANT = []
LISN = []
LISNT = []
NET = []
NETT = []

for i in range(10):

    s_star = create_true_source_set(G, num_of_sources=seed_size)
    contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=s_star)
    contagion.forward(actual_time_step_size)
    obs = contagion.get_infected_subgraph(step=actual_time_step_size-1)
    c_star = list(obs.nodes)
    
    start = time.time()
    gpsi_cs_kmeans = GPSI_cluster_sampling(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size, number_of_clusters=20)
    end = time.time()
    CST.append(end - start)
    dist = node_set_distance(gpsi_cs_kmeans, s_star, G)
    GPCS.append(dist)

    start = time.time()
    gpsi_ft = GPSI_ft(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size)
    end = time.time()
    RST.append(end - start)
    dist = node_set_distance(gpsi_ft, s_star, G)
    GPRS.append(dist)

    start = time.time()
    gpsi_vanilla = GPSI_vanilla(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size)
    end = time.time()
    SIT.append(end - start)
    dist = node_set_distance(gpsi_vanilla, s_star, G)
    GPSI.append(dist)

    start = time.time()
    jordan = cosasi.source_inference.multiple_source.fast_multisource_jordan_centrality(obs, G, 3).topn(1)
    end = time.time()
    jordan_result = []
    for j in jordan[0]:
        jordan_result.append(j)
    dist = node_set_distance(jordan_result, s_star, G)
    JORDANT.append(end - start)
    JORDAN.append(dist)

    start = time.time()
    lisn = cosasi.source_inference.multiple_source.fast_multisource_lisn(obs, G, actual_time_step_size, 3).topn((1))
    end = time.time()
    lisn_result = []
    for j in lisn[0]:
        lisn_result.append(j)
    dist = node_set_distance(lisn_result, s_star, G)
    LISNT.append(end - start)
    LISN.append(dist)

    start = time.time()
    sleuth = cosasi.source_inference.multiple_source.fast_multisource_netsleuth(obs, G, number_sources=3).topn((1))
    end = time.time()
    sleuth_result = []
    for j in sleuth[0]:
        sleuth_result.append(j)
    dist = node_set_distance(sleuth_result, s_star, G)
    NETT.append(end - start)
    NET.append(dist)

print('GPSI with fourier transfer and kmeans cluster sampling')
print('time: ', s.mean(CST), "+-", s.stdev(CST))
print('eval: ', s.mean(GPCS), '+-', s.stdev(GPCS))

print('GPSI with fourier transfer and random sampling')
print('time: ', s.mean(RST), "+-", s.stdev(RST))
print('eval: ', s.mean(GPRS), '+-', s.stdev(GPRS))

print('GPSI with vanilla sampling')
print('time: ', s.mean(SIT), "+-", s.stdev(SIT))
print('eval: ', s.mean(GPSI), '+-', s.stdev(GPSI))

print('Jordan sampling')
print('time: ', s.mean(JORDANT), "+-", s.stdev(JORDANT))
print('eval: ', s.mean(JORDAN), '+-', s.stdev(JORDAN))

print('LISN sampling')
print('time: ', s.mean(LISNT), "+-", s.stdev(LISNT))
print('eval: ', s.mean(LISN), '+-', s.stdev(LISN))

print('NET sampling')
print('time: ', s.mean(NETT), "+-", s.stdev(NETT))
print('eval: ', s.mean(NET), '+-', s.stdev(NET))

print('Pubmed')
G = PubMed()

GPCS = []
CST = []
GPRS = []
RST = []
GPSI = []
SIT = []
JORDAN = []
JORDANT = []
LISN = []
LISNT = []
NET = []
NETT = []

for i in range(10):

    s_star = create_true_source_set(G, num_of_sources=seed_size)
    contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=s_star)
    contagion.forward(actual_time_step_size)
    obs = contagion.get_infected_subgraph(step=actual_time_step_size-1)
    c_star = list(obs.nodes)
    
    start = time.time()
    gpsi_cs_kmeans = GPSI_cluster_sampling(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size, number_of_clusters=20)
    end = time.time()
    CST.append(end - start)
    dist = node_set_distance(gpsi_cs_kmeans, s_star, G)
    GPCS.append(dist)

    start = time.time()
    gpsi_ft = GPSI_ft(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size)
    end = time.time()
    RST.append(end - start)
    dist = node_set_distance(gpsi_ft, s_star, G)
    GPRS.append(dist)

    start = time.time()
    gpsi_vanilla = GPSI_vanilla(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, seed_size)
    end = time.time()
    SIT.append(end - start)
    dist = node_set_distance(gpsi_vanilla, s_star, G)
    GPSI.append(dist)

    start = time.time()
    jordan = cosasi.source_inference.multiple_source.fast_multisource_jordan_centrality(obs, G, 3).topn(1)
    end = time.time()
    jordan_result = []
    for j in jordan[0]:
        jordan_result.append(j)
    dist = node_set_distance(jordan_result, s_star, G)
    JORDANT.append(end - start)
    JORDAN.append(dist)

    start = time.time()
    lisn = cosasi.source_inference.multiple_source.fast_multisource_lisn(obs, G, actual_time_step_size, 3).topn((1))
    end = time.time()
    lisn_result = []
    for j in lisn[0]:
        lisn_result.append(j)
    dist = node_set_distance(lisn_result, s_star, G)
    LISNT.append(end - start)
    LISN.append(dist)

    start = time.time()
    sleuth = cosasi.source_inference.multiple_source.fast_multisource_netsleuth(obs, G, number_sources=3).topn((1))
    end = time.time()
    sleuth_result = []
    for j in sleuth[0]:
        sleuth_result.append(j)
    dist = node_set_distance(sleuth_result, s_star, G)
    NETT.append(end - start)
    NET.append(dist)

print('GPSI with fourier transfer and kmeans cluster sampling')
print('time: ', s.mean(CST), "+-", s.stdev(CST))
print('eval: ', s.mean(GPCS), '+-', s.stdev(GPCS))

print('GPSI with fourier transfer and random sampling')
print('time: ', s.mean(RST), "+-", s.stdev(RST))
print('eval: ', s.mean(GPRS), '+-', s.stdev(GPRS))

print('GPSI with vanilla sampling')
print('time: ', s.mean(SIT), "+-", s.stdev(SIT))
print('eval: ', s.mean(GPSI), '+-', s.stdev(GPSI))

print('Jordan sampling')
print('time: ', s.mean(JORDANT), "+-", s.stdev(JORDANT))
print('eval: ', s.mean(JORDAN), '+-', s.stdev(JORDAN))

print('LISN sampling')
print('time: ', s.mean(LISNT), "+-", s.stdev(LISNT))
print('eval: ', s.mean(LISN), '+-', s.stdev(LISN))

print('NET sampling')
print('time: ', s.mean(NETT), "+-", s.stdev(NETT))
print('eval: ', s.mean(NET), '+-', s.stdev(NET))