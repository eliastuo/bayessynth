def summarize_factors(target, data, trace_df, m,
                      years=None, cwd=None, dst=None, replace_files=False,
                      quantiles=[0.0, 0.005, 0.025, 0.5, 0.975, 0.995, 1.0]):
    """Analyze an MCMC trace: derive the latent factor distributions.
    
    This function summarizes a trace dataframe into
    a description of the mean and various probability intervals
    of the latent factor trajectories of.

    Positional arguments:
        target - (str) title of target society in df's columns
        data - (pandas dataframe) dataset originally analyzed
        trace_df - (pandas dataframe) sampling output for the appropriate variables
        m - (int) number of factors

    Optional keyword arguments:
        years - (numpy array) subset of data's index; full index by default
        cwd - (str) current working directory path; inferred by default
        dst - (str) filepath where to save function outputs; when not specified, nothing saved on disk
        replace_files - (bool) overwrite pre-existing output files; False by default
        quantiles - (list/numpy array) quantiles to capture; range, 99%-CI, 95%-CI, and median by default

    Output: pandas dataframe
    
    """
    # Protect trace object from side effects
    trace_df = trace_df.copy()

    # Parameterize country selection & time period
    years = years if years else data.index.values
    start, end = years[0], years[-1] + 1
    n = len(years)

    # Other parameterization
    cwd = cwd if cwd else Path(__file__).parent.resolve()
    replacements = ((0.0, 'min'), (1.0, 'max'), (0.5, 'median'))

    # Initialize storage
    dflist = []
    
    # Iterate over factors
    for i in range(m):
        # Carve out data for the i-th factor
        mslice = slice(i, m*n, m)
        df_m = trace_df.iloc[:, mslice]
        # Calculate summary statistics
        mean = df_m.mean()
        ci_list = [df_m.quantile(quantile) for quantile in quantiles]
        # Combine statistics in a single dataframe
        statistics = [mean] + ci_list
        summary = pd.concat(statistics, axis=1)
        # Give each statistic a title
        quantile_names = ['ci_' + str(ci) for ci in quantiles]
        stat_names = ['mean'] + quantile_names
        # Give intuitive names to a few special statistics (min, max, median)
        for number, word in replacements:
            number_name = 'ci_' + str(number)
            if number_name in stat_names:
                index = stat_names.index(number_name)
                stat_names[index] = word
        # Name columns and rows of the summary df, and append it to storage
        summary.columns = stat_names
        summary.index = years
        dflist.append(summary)

    # Optionally, save summary on disk

    if dst:
        if dst == 'default':
            dst = os.path.join(cwd, 'BSCAnalysis', 'L', str(m) + target, 'factor_summary.pkl')
        dstdir = os.path.split(dst)[0]
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        elif os.path.exists(dst) and not replace_files:
            msg = "Summary file already exists. Switch 'replace_files=True'."
            raise Exception(msg)
        with open(dst, 'wb') as file:
            pickle.dump(dflist, file)

    # Return the list of dataframes

    return dflist

