def summarize_ppc(target, data, trace_df, m,
                  years=None, cwd=None, dst=None, sigma_colname='sigma', replace_files=False, include_observed=True,
                  quantiles=[0.0, 0.005, 0.025, 0.5, 0.975, 0.995, 1.0],
                  other_columns=None):
    """Analyze an MCMC trace: derive the untreated posterior predictive trajectory.
    
    This function summarizes a trace dataframe into
    a description of the mean and various probability intervals
    of the posterior predictive trajectory of the target society.

    Positional arguments:
        target - (str) title of target society in df's columns
        data - (pandas dataframe) dataset originally analyzed
        trace_df - (pandas dataframe) sampling output for the appropriate variables
        m - (int) number of factors

    Optional keyword arguments:
        years - (numpy array) subset of data's index; full index by default
        cwd - (str) current working directory path; inferred by default
        dst - (str) filepath where to save function outputs; when not specified, nothing saved on disk
        sigma_colname - (str) name of the noise standard deviation varaible in trace_df
        replace_files - (bool) overwrite pre-existing output files; False by default
        include_observed - (bool) include observed data in the output df; True by default
        quantiles - (list/numpy array) quantiles to capture; range, 99%-CI, 95%-CI, and median by default
        other_columns - (iterable of (str, array)) additional columns to include

    Output: pandas dataframe
    
    """
    # Protect trace object from side effects
    trace_df = trace_df.copy()

    # Parameterize country selection & time period
    all_countries = data.columns.to_list()
    years = years if years else data.index.values
    start, end = years[0], years[-1] + 1
    n = len(years)
    order = all_countries.index(target)

    # Other parameterization
    cwd = cwd if cwd else Path(__file__).parent.resolve()
    varnames = trace_df.columns.to_list()
    sigma_position = varnames.index(sigma_colname)
    varnames_short = varnames[:sigma_position] + varnames[sigma_position+1:]
    replacements = ((0.0, 'min'), (1.0, 'max'), (0.5, 'median'))

    # Convert from posterior to predictive posterior

    # Generate noise
    chainlen = trace_df.shape[0]
    sigmas = trace_df[sigma_colname].values
    generate_noise = lambda i: np.random.normal(loc=0.0, scale=sigmas[i], size=n)
    noise_vectors = [generate_noise(i) for i in range(chainlen)]
    noise = np.array(noise_vectors)

    # Combine noise and posterior values
    df = trace_df[varnames_short]
    df = df + noise
    df.columns = years

    # Record summary statistics

    # Calculate summary stats
    mean = df.mean()
    ci_list = [df.quantile(quantile) for quantile in quantiles]

    # Capture summary statistics into a new dataframe
    statistics = [mean] + ci_list
    summary = pd.concat(statistics, axis=1)

    # Generate names for the summary statistic variables
    quantile_names = ['ci_' + str(ci) for ci in quantiles]
    stat_names = ['mean'] + quantile_names
    for number, word in replacements:
        number_name = 'ci_' + str(number)
        if number_name in stat_names:
            index = stat_names.index(number_name)
            stat_names[index] = word
    summary.columns = stat_names

    # Include observed data
    if include_observed:
        summary['observed'] = data[target]

    # Include other specified columns
    if other_columns:
        try:
            for name, vector in other_columns:
                summary[name] = vector
        except TypeError:
            msg = "'other_columns' must contain (str, array) pairs"
            raise TypeError(msg)

    # Optionally, save summary on disk

    if dst:
        if dst == 'default':
            dst = os.path.join(cwd, 'BSCAnalysis', 'L', str(m) + target, 'ppc_summary.pkl')
        dstdir = os.path.split(dst)[0]
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        elif os.path.exists(dst) and not replace_files:
            msg = "Summary file already exists. Switch 'replace_files=True'."
            raise Exception(msg)
        with open(dst, 'wb') as file:
            pickle.dump(summary, file)

    # Return a dataframe

    return summary

