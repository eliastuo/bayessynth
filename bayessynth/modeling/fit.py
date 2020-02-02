def fit(df, target, cutoff, prior,
        years=None, countries=None, llambda_raw=2, m=4,
        cwd=None, out_dst=None,
        chains=2, chain_length=25000, tune_length=5000, target_accept=0.90, treedepth=12, samplesize=2000, progressbar=True, mcmc_kwargs={},
        replace_files=False, verbose=True, trace_only=False):
    """Main function to estimate the BSC model for a single dataset.

    The BSC model is fitted on the dataset provided,
    using th pymc3 implementation of the NUTS (MCMC) sampler.
    The resulting sample trace is stored on hard drive
    along with various other summary statistics.
    
    Positional arguments:
        df - (pandas dataframe) data to be analyzed
        target - (str) title of target society in df's columns
        cutoff - (int) the start year of treatment for target society
        prior - (dict) prior distribution parameters

    Optional keyword arguments:
        years - (numpy array) subset of df's index; full index by default
        countries - (list) subset of countries in df's columns; all columns by default
        llambda_raw - (int) scaling parameter for factor prior standard devations; 2 by default
        m - (int) number of factors; 4 by default
        cwd - (str) current working directory path; inferred by default
        out_dst - (str) destination path for sampler output; automatically constructed by default
        chains - (int) number of parallel MCMC chains; 2 by default
        chain_length - (int) number of proper sampler steps per chain; 25000 by default
        tune_length - (int) number of additional tune-in steps per chain; 5000 by default
        target_accept - (float) NUTS sampler target acceptance rate; 0.9 by default
        treedepth - (int) NUTS sampler maximum tree depth; 12 by default
        samplesize - (int) size of a separately saved representative mini-sample; 2000 by default
        progressbar - (bool) print a live progressbar while sampling; True by default
        mcmc_kwargs - (dict) additional keyword arguments for the NUTS sampler
        replace_files - (bool) overwrite pre-existing sampler output files; False by default
        verbose - (bool) print messages to user; True by default
        trace_only - (bool) save only sampling trace, no summaries; False by default

    Output: None
    """

    # Parameter setup

    # Set up time period
    years = years if years else df.index.values
    start, end = years[0], years[-1] + 1
    years_short = np.arange(start, cutoff)
    years_rest = np.arange(cutoff, end)
    n, n_short, n_rest = len(years), len(years_short), len(years_rest)
    # Set up countries
    countries = countries if countries else df.columns.tolist()
    countries = [country for country in countries if country != target]
    all_countries = countries + [target]
    d = len(all_countries)
    # File paths
    cwd = cwd if cwd else Path(__file__).parent.resolve()
    if not out_dst:
        out_dst = os.path.join(cwd, 'BSCOutput', 'L' + str(m) + target)
    csv = os.path.join(out_dst, 'trace')
    dst_results = os.path.join(out_dst, 'results.pkl')
    dst_modeldict = os.path.join(out_dst, 'modeldict.pkl')
    dst_smallsample = os.path.join(out_dst, 'smallsample.pkl')
    summary = {}
    # Create or replace directories
    for path in [out_dst]:
        if os.path.exists(path):
            if replace_files:
                shutil.rmtree(path)
            else:
                msg = 'Directory "{}" already exists.'.format(path)
                raise Exception(msg)
        os.makedirs(path)
    # Latent factor prior variance
    llambda = llambda_raw**2
    
    # Data structure setup

    # PCA components
    dh = df.copy()[all_countries]
    pca = PCA(n_components=m)
    pca.fit(dh[countries])
    database = pca.transform(dh[countries])
    database_flat = database.reshape(n*m)
    base_var = database.var(axis=0)
    factor_var = np.concatenate([base_var for i in range(n)])
    # Outcome data
    datarest = dh.loc[years, all_countries].values
    # Indicator matrix
    indicator = np.zeros((n, d))
    indicator[n_short:, -1] = 1
    D = indicator.T
    # Pre-reform dummy vector of target society treatment effects
    dummy = np.zeros(n_short)
    # Identity matrices
    identity_md = np.identity(m*d)
    identity_nm = np.identity(n*m)

    # Sampler parameterization
    mcmc_kwargs.update({'chains': chains,
                        'draws': chain_length,
                        'tune': tune_length,
                        'progressbar': progressbar,
                        'target_accept': target_accept,
                        'max_treedepth': treedepth})

    # Run MCMC sampling

    # Open a pymc3 model in context manager
    with pm.Model() as model:
        # Probabilistic model

        # Society fixed effect (K) hierarchical prior
        k_mu = pm.Normal('k_mu', mu=prior['k_mu'], sd=prior['k_sd'])
        k_sd = pm.HalfCauchy('k_sd', beta=prior['k_gamma'])
        k_offset = pm.Normal('k_offset', mu=0, sd=1, shape=d)
        kappa = (k_offset * k_sd) + k_mu
        # Factor loadings' (B) hierarchical prior
        b_mu = pm.Normal('b_mu', mu=prior['b_mu'], sd=prior['b_sd'], shape=m)
        b_sd = pm.HalfCauchy('b_sd', beta=prior['b_gamma'], shape=m)
        b_long = pm.MvNormal('b_long', mu=0, cov=identity_md, shape=m*d)
        b_offset = b_long.reshape((m, d)).T
        beta = ((b_offset * b_sd) + b_mu).T
        # Noise standard deviation prior
        sigma = pm.HalfCauchy('sigma', beta=prior['sigma_gamma'])
        # Latent factor prior
        L_long = pm.MvNormal('L_long', mu=database_flat,
                             cov=identity_nm*factor_var*llambda, shape=n*m)
        L = L_long.reshape((n, m))
        # Year fixed effect prior
        delta_raw = pm.Normal('delta_raw', mu=prior['delta_mu'], sd=prior['delta_sd'], shape=(n, 1))
        delta = delta_raw - delta_raw.mean()
        # Treatment effect prior
        alpha = pm.Normal('alpha', mu=prior['alpha_mu'], sd=prior['alpha_sd'], shape=n_rest)
        alpha_vector = T.concatenate([dummy, alpha])
        alpha_matrix = (D * alpha_vector).T
        # Untreated state
        synthetic = pm.Deterministic('synthetic', T.dot(L, beta) + kappa + delta)
        # Likelihood
        y = pm.Normal('y', mu=synthetic+alpha_matrix, sd=sigma, observed=datarest)

        # MCMC sampling

        # Initialize hard drive storage for sampler
        db = pm.backends.Text(csv)
        # Draw a sample
        trace = pm.sample(**mcmc_kwargs, trace=db)
    
    if not trace_only:

        # Format and save output
        
        # Save model and trace objects
        with open(dst_modeldict, 'wb') as file:
            pickle.dump({'model': model, 'trace': trace}, file)
        # Save a small random sample separately
        samples = np.random.choice(trace, size=samplesize)
        with open(dst_smallsample, 'wb') as file:
            pickle.dump(samples, file)

        # Calculate summary statistics

        # Record observed data and initialize storage variables
        observed = datarest[:, -1]
        greater = np.zeros(n)
        difference = np.zeros(n)
        # Loop over the small sample
        for draw in samples:
            # Record untreated state and resample noise
            mu = draw['synthetic'][:, -1]
            sigma = draw['sigma']
            noise = np.random.normal(loc=0.0, scale=sigma, size=n)
            # Construct draw from posterior predictive and compare to observed data
            prediction = mu + noise
            greater += (observed >= prediction)
            difference += (observed - prediction)
        # Calculate the average difference to observed data
        difference /= samplesize
        # Calculate the share of predictions less than observed data
        greater /= samplesize
        # Calculate (Gelman-Rubin) diagnostic of sampler convergence
        rubin = pm.diagnostics.gelman_rubin(trace, varnames=['alpha'])['alpha']
        # Save summary statistics
        results = [greater, difference, rubin]
        with open(dst_results, 'wb') as file:
            pickle.dump(results, file)

    # Report back to user
    if verbose:
        print("Finished fitting the model.")
        print("The following output files were generated:")
        msg = "\t{}"
        for path in [dst_modeldict, csv, dst_smallsample, dst_results]:
            print(msg.format(path))

    return None

