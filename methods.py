import torch

from my_util import *

################################################
# BOSI with singletaskGP, no modification
################################################

def GPSI1(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model):
    # initialize the GP model with several (c, s) pairs
    candidates = create_candidate_pool(G, c_star, candidate_size)

    train_X = []
    train_Y = []
    source_sets = []
    for estimated_source_number in range(2, 6):

        for i in range(5):
            source_set = sample_from_candidate_pool(candidates, estimated_source_number)

            contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=source_set, recovery_rate=recovery_rate)

            peak_mean, peak_var = source_coverage(contagion, c_star, num_of_sims)

            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)
            input = torch.tensor(input)

            train_X.append(input)
            train_Y.append([peak_mean])
            source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # model = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        # fit_fully_bayesian_model_nuts(model)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 100 random samples, evaluate similarity with existing samples, pick 20 most different ones, optimize EI
        ################################################
        combs = []

        for estimated_source_number in range(2, 6):

            for i in range(5):
                source_set = sample_from_candidate_pool(candidates, estimated_source_number)
                input = []
                for item in candidates:
                    if item in source_set:
                        input.append(1)
                    else:
                        input.append(0)

                input = torch.tensor([input])
                combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=selected, recovery_rate=recovery_rate)
        new_Y, var = source_coverage(contagion, c_star, num_of_sims)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

################################################
# BOSI with singletaskGP, last round evaluate all
################################################

def GPSI2(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model):
    # initialize the GP model with several (c, s) pairs
    candidates = create_candidate_pool(G, c_star, candidate_size)

    train_X = []
    train_Y = []
    source_sets = []
    for estimated_source_number in range(2, 6):

        for i in range(5):
            source_set = sample_from_candidate_pool(candidates, estimated_source_number)

            contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=source_set, recovery_rate=recovery_rate)

            peak_mean, peak_var = source_coverage(contagion, c_star, num_of_sims)

            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)
            input = torch.tensor(input)

            train_X.append(input)
            train_Y.append([peak_mean])
            source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # model = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        # fit_fully_bayesian_model_nuts(model)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 100 random samples, evaluate similarity with existing samples, pick 20 most different ones, optimize EI
        ################################################
        combs = []

        for estimated_source_number in range(2, 6):

            for i in range(5):
                source_set = sample_from_candidate_pool(candidates, estimated_source_number)
                input = []
                for item in candidates:
                    if item in source_set:
                        input.append(1)
                    else:
                        input.append(0)

                input = torch.tensor([input])
                combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=selected, recovery_rate=recovery_rate)
        new_Y, var = source_coverage(contagion, c_star, num_of_sims)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    best = float('-inf')
    best_source = None

    for estimated_source_number in range(2, 6):
        for source_set in combinations(candidates, estimated_source_number):
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)
            input = torch.FloatTensor([input])
            y_pred = model(input).loc
            if y_pred > best:
                best = y_pred
                best_source = source_set

    return best_source

################################################
# BOSI with singletaskGP, graph sampling
################################################
def GPSI_with_graph_sampling(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model):
    # initialize the GP model with several (c, s) pairs
    candidates = create_candidate_pool(G, c_star, candidate_size)

    train_X = []
    train_Y = []
    source_sets = []
    for estimated_source_number in range(2, 6):

        for i in range(5):
            source_set = sample_from_candidate_pool(candidates, estimated_source_number)

            contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=source_set, recovery_rate=recovery_rate)

            peak_mean, peak_var = source_coverage(contagion, c_star, num_of_sims)

            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)
            input = torch.tensor(input)

            train_X.append(input)
            train_Y.append([peak_mean])
            source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = SingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        # model = SaasFullyBayesianSingleTaskGP(train_X, train_Y)
        # fit_fully_bayesian_model_nuts(model)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 100 random samples, evaluate similarity with existing samples, pick 20 most different ones, optimize EI
        ################################################
        combs = []
        sets = []
        scores = []

        for estimated_source_number in range(2, 6):

            for i in range(25):
                source_set = sample_from_candidate_pool(candidates, estimated_source_number)
                score = distance_sampling(source_set, source_sets, G)

                input = []
                for item in candidates:
                    if item in source_set:
                        input.append(1)
                    else:
                        input.append(0)

                sets.append(input)
                scores.append(score)

        sets = np.array(sets)
        scores = np.array(scores)
        indices = np.argpartition(scores, -20)[-20:]
        top20 = sets[indices]

        for input in top20:

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=selected, recovery_rate=recovery_rate)
        new_Y, var = source_coverage(contagion, c_star, num_of_sims)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

################################################
# BOSI with GCN, no modification
################################################
def GCNSI(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model):
    candidates = create_candidate_pool(G, c_star, candidate_size)

    train_X = []
    train_Y = []
    source_sets = []
    for estimated_source_number in range(2, 6):

        for i in range(5):
            source_set = sample_from_candidate_pool(candidates, estimated_source_number)

            contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=source_set, recovery_rate=recovery_rate)

            peak_mean, peak_var = source_coverage(contagion, c_star, num_of_sims)

            input = []
            for item in G.nodes:
                if item in source_set:
                    input.append([1])
                else:
                    input.append([0])
            input = torch.FloatTensor(input)

            train_X.append(input)
            train_Y.append([peak_mean])
            source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    ################################################
    # Create edge-list from nx graph
    ################################################

    start = []
    end = []
    for edge in G.edges():
        start.append(edge[0])
        end.append(edge[1])
        start.append(edge[1])
        end.append(edge[0])
    edges = [start, end]
    edges = torch.tensor(edges)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = regGCN()
        optimizer = torch.optim.Adam([dict(params=model.conv1.parameters(), weight_decay=5e-4),
                                      dict(params=model.conv2.parameters(), weight_decay=0)], lr=0.01)
        model.train()
        for index in range(len(train_X)):
            x = train_X[index]
            y = train_Y[index].squeeze()
            optimizer.zero_grad()
            out = model(x, edges).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()

        combs = []

        for estimated_source_number in range(2, 6):

            for i in range(5):
                source_set = sample_from_candidate_pool(candidates, estimated_source_number)
            # for source_set in combinations(candidates, estimated_source_number):
                input = []
                for item in G.nodes:
                    if item in source_set:
                        input.append([1])
                    else:
                        input.append([0])

                input = torch.FloatTensor([input])
                combs.append(input)

        combs = torch.stack(combs)

        candidate = None
        acq_value = float('-inf')

        for comb in combs:
            y_pred = model(comb, edges)
            if y_pred > acq_value:
                acq_value = y_pred
                candidate = comb

        selected = []
        for i in range(len(G.nodes)):

            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(list(G.nodes)[i])

        contagion = Contagion(G=G, model='si', infection_rate=infect_rate, source=selected, recovery_rate=recovery_rate)
        new_Y, var = source_coverage(contagion, c_star)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(len(G.nodes)):
        if int(s[i]) == 1:
            result.append(list(G.nodes)[i])

    return result

################################################
# BOSI with GCN, last round evaluate all
################################################
def GCNSI2(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model):
    candidates = create_candidate_pool(G, c_star, candidate_size)

    train_X = []
    train_Y = []
    source_sets = []
    for estimated_source_number in range(2, 6):

        for i in range(5):
            source_set = sample_from_candidate_pool(candidates, estimated_source_number)

            contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=source_set, recovery_rate=recovery_rate)

            peak_mean, peak_var = source_coverage(contagion, c_star, num_of_sims)

            input = []
            for item in G.nodes:
                if item in source_set:
                    input.append([1])
                else:
                    input.append([0])
            input = torch.FloatTensor(input)

            train_X.append(input)
            train_Y.append([peak_mean])
            source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    ################################################
    # Create edge-list from nx graph
    ################################################

    start = []
    end = []
    for edge in G.edges():
        start.append(edge[0])
        end.append(edge[1])
        start.append(edge[1])
        end.append(edge[0])
    edges = [start, end]
    edges = torch.tensor(edges)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = regGCN()
        optimizer = torch.optim.Adam([dict(params=model.conv1.parameters(), weight_decay=5e-4),
                                      dict(params=model.conv2.parameters(), weight_decay=0)], lr=0.01)
        model.train()
        for index in range(len(train_X)):
            x = train_X[index]
            y = train_Y[index].squeeze()
            optimizer.zero_grad()
            out = model(x, edges).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()

        combs = []

        for estimated_source_number in range(2, 6):

            for i in range(5):
                source_set = sample_from_candidate_pool(candidates, estimated_source_number)
            # for source_set in combinations(candidates, estimated_source_number):
                input = []
                for item in G.nodes:
                    if item in source_set:
                        input.append([1])
                    else:
                        input.append([0])

                input = torch.FloatTensor([input])
                combs.append(input)

        combs = torch.stack(combs)

        candidate = None
        acq_value = float('-inf')

        for comb in combs:
            y_pred = model(comb, edges)
            if y_pred > acq_value:
                acq_value = y_pred
                candidate = comb

        selected = []
        for i in range(len(G.nodes)):

            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(list(G.nodes)[i])

        contagion = Contagion(G=G, model='si', infection_rate=infect_rate, source=selected, recovery_rate=recovery_rate)
        new_Y, var = source_coverage(contagion, c_star)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    best = float('-inf')
    best_source = None

    for estimated_source_number in range(2, 6):
        for source_set in combinations(candidates, estimated_source_number):
            input = []
            for item in G.nodes:
                if item in source_set:
                    input.append([1])
                else:
                    input.append([0])
            input = torch.FloatTensor([input])
            y_pred = model(input, edges)
            if y_pred > best:
                best = y_pred
                best_source = source_set

    return best_source

################################################
# BOSI with GCN, fourier sampling
################################################
def GCNSI_fs(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model):

    normalized_laplacian = nx.normalized_laplacian_matrix(G)
    _, UT = np.linalg.eigh(normalized_laplacian.todense())
    candidates = create_candidate_pool(G, c_star, candidate_size)

    train_X = []
    train_Y = []
    source_sets = []
    train_X_fourier = []
    for estimated_source_number in range(2, 6):

        for i in range(5):
            source_set = sample_from_candidate_pool(candidates, estimated_source_number)

            contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=source_set, recovery_rate=recovery_rate)

            peak_mean, peak_var = source_coverage(contagion, c_star, num_of_sims)

            input = []
            input_for_fourier = []
            for item in G.nodes:
                if item in source_set:
                    input.append([1])
                    input_for_fourier.append(1)
                else:
                    input.append([0])
                    input_for_fourier.append(0)

            X_fourier = np.matmul(input_for_fourier, UT)
            train_X_fourier.append(X_fourier)

            input = torch.FloatTensor(input)
            train_X.append(input)
            train_Y.append([peak_mean])

            source_sets.append(source_set)

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    ################################################
    # Create edge-list from nx graph
    ################################################

    start = []
    end = []
    for edge in G.edges():
        start.append(edge[0])
        end.append(edge[1])
        start.append(edge[1])
        end.append(edge[0])
    edges = [start, end]
    edges = torch.tensor(edges)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = regGCN()
        optimizer = torch.optim.Adam([dict(params=model.conv1.parameters(), weight_decay=5e-4),
                                      dict(params=model.conv2.parameters(), weight_decay=0)], lr=0.01)
        model.train()
        for index in range(len(train_X)):
            x = train_X[index]
            y = train_Y[index].squeeze()
            optimizer.zero_grad()
            out = model(x, edges).squeeze()
            loss = F.mse_loss(out, y)
            loss.backward()
            optimizer.step()

        candidate_source_sets = []

        for estimated_source_number in range(2, 6):

            for i in range(25):
                source_set = sample_from_candidate_pool(candidates, estimated_source_number)
                candidate_source_sets.append(source_set)

        final_source_sets = fourier_sampler(G, candidate_source_sets, train_X_fourier, UT, 20)

        combs = []

        for source_set in final_source_sets:

                input = []
                for item in G.nodes:
                    if item in source_set:
                        input.append([1])
                    else:
                        input.append([0])

                input = torch.FloatTensor([input])
                combs.append(input)

        combs = torch.stack(combs)

        candidate = None
        acq_value = float('-inf')

        for comb in combs:
            y_pred = model(comb, edges)
            if y_pred > acq_value:
                acq_value = y_pred
                candidate = comb

        candidate_fourier = np.matmul(candidate.squeeze().numpy(), UT)

        selected = []
        for i in range(len(G.nodes)):

            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(list(G.nodes)[i])

        contagion = Contagion(G=G, model='si', infection_rate=infect_rate, source=selected, recovery_rate=recovery_rate)
        new_Y, var = source_coverage(contagion, c_star)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)  # Add a new dimension for the new evaluation
        train_X_fourier.append(candidate_fourier)

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(len(G.nodes)):
        if int(s[i]) == 1:
            result.append(list(G.nodes)[i])

    return result

################################################
# 7-20
################################################

# Gaussian Process with Fourier Transfer, sampling through clustering， RBF kernel
def GPSI_cluster_sampling(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, number_of_sources, number_of_clusters):

    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect


    if diffusion_model == 'si':
        candidates = create_candidate_pool(G, c_star, candidate_size)
    elif diffusion_model == 'sir' or 'sis':
        candidates = create_candidate_pool_from_whole_graph(G, candidate_size)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidates,number_of_sources,UT)

    kmeans = KMeans(n_clusters=number_of_clusters, random_state=0).fit(sets_after_fourier_transfer)
    labels = kmeans.labels_

    groups = [[] for i in range(number_of_clusters)]

    for j in range(len(labels)):
        groups[labels[j]].append(sets_after_fourier_transfer[j])

    train_X = []
    train_Y = []

    # initialize the BO with an instance from each cluster
    for i in range(number_of_clusters):

        selected_signal = random.choice(groups[i])
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)
        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=source_set, recovery_rate=recovery_rate)
        peak_mean, peak_var = source_coverage(contagion, c_star, num_of_sims)
        input = torch.FloatTensor(selected_signal)
        train_X.append(input)
        train_Y.append([peak_mean])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):

        # from each cluster, sample 1 instances, select the one with the highest acquisition function value from the samples
        inputs= []
        for i in range(number_of_clusters):
            samples = random.sample(groups[i], 1)
            for sample in samples:
                inputs.append(torch.FloatTensor(sample))

        inputs = torch.stack(inputs).type(torch.float)

    # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = inputs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=selected, recovery_rate=recovery_rate)
        new_Y, var = source_coverage(contagion, c_star, num_of_sims)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    # final_index = sets_after_fourier_transfer.index(s)
    identified_set = find_source_set_from_fourier(s, number_of_sources, UT_inv)

    return identified_set

# Gaussian Process with node selection, random sampling

def GPSI_vanilla(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, number_of_sources):
    # initialize the GP model with several (c, s) pairs
    if diffusion_model == 'si':
        candidates = create_candidate_pool(G, c_star, candidate_size)
    elif diffusion_model == 'sir' or 'sis':
        candidates = create_candidate_pool_from_whole_graph(G, candidate_size)

    train_X = []
    train_Y = []
    source_sets = []

    for i in range(20):
        source_set = sample_from_candidate_pool(candidates, number_of_sources)

        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=source_set, recovery_rate=recovery_rate)

        peak_mean, peak_var = source_coverage(contagion, c_star, num_of_sims)

        input = []
        for item in candidates:
            if item in source_set:
                input.append(1)
            else:
                input.append(0)
        input = torch.tensor(input)

        train_X.append(input)
        train_Y.append([peak_mean])
        source_sets.append(source_set)

    train_X = torch.stack(train_X).type(torch.double)
    train_Y = torch.tensor(train_Y).type(torch.double)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 20 random samples
        ################################################
        combs = []

        for i in range(20):
            source_set = sample_from_candidate_pool(candidates, number_of_sources)
            input = []
            for item in candidates:
                if item in source_set:
                    input.append(1)
                else:
                    input.append(0)

            input = torch.tensor([input])
            combs.append(input)

        combs = torch.stack(combs).type(torch.double)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        selected = []
        for i in range(candidate_size):
            if list(candidate.squeeze().numpy())[i] == 1:
                selected.append(candidates[i])

        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=selected, recovery_rate=recovery_rate)
        new_Y, var = source_coverage(contagion, c_star, num_of_sims)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate[0]], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1,1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())
        source_sets.append(selected)

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    result = []
    for i in range(candidate_size):
        if int(s[i]) == 1:
            result.append(candidates[i])

    return result

# Gaussian Process with fourier transfer, random sampling

def GPSI_ft(G, c_star, num_iterations, num_of_sims, infect_rate, candidate_size, diffusion_model, number_of_sources):
    # initialize the GP model with several (c, s) pairs

    nl = nx.normalized_laplacian_matrix(G)
    _, eig_vect = np.linalg.eigh(nl.todense())
    UT = np.linalg.inv(eig_vect)
    UT_inv = eig_vect

    if diffusion_model == 'si':
        candidates = create_candidate_pool(G, c_star, candidate_size)
    elif diffusion_model == 'sir' or 'sis':
        candidates = create_candidate_pool_from_whole_graph(G, candidate_size)

    sets_after_fourier_transfer = fourier_transfer_for_all_candidate_set(candidates,number_of_sources,UT)

    train_X = []
    train_Y = []

    initial_indices = random.sample(range(len(sets_after_fourier_transfer)), 20)

    for index in initial_indices:
        selected_signal = sets_after_fourier_transfer[index]
        source_set = find_source_set_from_fourier(selected_signal, number_of_sources, UT_inv)

        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=source_set, recovery_rate=recovery_rate)

        peak_mean, peak_var = source_coverage(contagion, c_star, num_of_sims)

        input = torch.FloatTensor(selected_signal)

        train_X.append(input)
        train_Y.append([peak_mean])

    train_X = torch.stack(train_X)
    train_Y = torch.tensor(train_Y)

    function_values = [train_Y.max().item()]
    acquisition_values = []
    max_train_Y_values = [train_Y.max().item()]

    for iteration in range(num_iterations):
        # Fit a single-output GP model to the observed data
        model = RBFSingleTaskGP(train_X, train_Y)
        mll = ExactMarginalLogLikelihood(model.likelihood, model)
        fit_gpytorch_model(mll)

        acq_func = ExpectedImprovement(model=model, best_f=train_Y.max())

        ################################################
        # 20 random samples
        ################################################
        combs = []

        samples = random.sample(sets_after_fourier_transfer, 20)

        for sample in samples:
            input = torch.FloatTensor(sample)
            combs.append(input)

        combs = torch.stack(combs).type(torch.float)

        candidate, acq_value = optimize_acqf_discrete(
            acq_function=acq_func,
            q=1,                                                # Number of candidates to sample in each iteration
            choices = combs)

        found_candidate = candidate[0]

        signal = found_candidate.tolist()

        selected = find_source_set_from_fourier(signal, number_of_sources, UT_inv)

        contagion = Contagion(G=G, model=diffusion_model, infection_rate=infect_rate, source=selected, recovery_rate=recovery_rate)
        new_Y, var = source_coverage(contagion, c_star, num_of_sims)

        # Update the observed data with the new evaluation
        train_X = torch.cat([train_X, candidate], dim=0)
        train_Y = torch.cat([train_Y, new_Y.resize(1, 1)], dim=0)  # Add a new dimension for the new evaluation

        # Store function value, acquisition function value, and maximum value of train_Y
        function_values.append(new_Y.item())
        acquisition_values.append(acq_value.item())
        max_train_Y_values.append(train_Y.max().item())

    index = [index for index, item in enumerate(train_Y) if item == train_Y.max()]
    s = list(train_X[index[0]])
    # final_index = sets_after_fourier_transfer.index(s)
    identified_set = find_source_set_from_fourier(s, number_of_sources, UT_inv)

    return identified_set