def plot_ppc_bell(tracedf, data, target, m,
                  difference=True, samplesize='full', year_pos=None, sigma_varname='sigma', dstdir=None, filename=None, alpha=0.333,
                  line_kde={'color': 'k', 'label': '', 'linewidth': 1.3},
                  shade_kde={'color': 'tab:blue', 'label': '', 'shade': True},
                  line_obs={'linestyle': '--', 'color': 'k'},
                  bandwidth='scott', xlabel='Value', ylabel='Probability density', title=None, image_format='pdf', output='disk',
                  verbose=True):
    """Visualize the target society's cumulative treatment effect.
    
    This function turns a trace dataframe
    (with appropriate variables present)
    into a visualization of the posterior predictive distribution
    of the size of the cumulative treatment effect for the target society
    at a specified year (end of time period by default).

    Positional arguments:
        tracedf - (pandas dataframe) sampling output for the appropriate variables
        data - (pandas dataframe) dataset originally analyzed
        target - (str) title of target society in df's columns
        m - (int) number of factors; 4 by default

    Optional keyword arguments:
        difference - (bool) visualize treatment effect instead of absolute ppc value; True by default
        samplsize - (str or int) 'full' to use the full trace (default); integer to select ransom subsample
        year_pos - (int) position of the target year within  data's index (-1 ie. last year by default)
        sigma_varname - (str) name of the noise standard deviation varaible in tracedf; 'sigma' by default
        dstdir - (str) destination path for plot output directory; automatically constructed by default
        filename - (str) destination path for plot output filename; automatically constructed by default 
        alpha - (float) shading transparency of the confidence interval; 0.333 by default
        line_kde - (dict) matplotlib kwargs for formatting density plot's outline; preconfigured by default
        shade_kde - (dict) matplotlib kwargs for formatting density plot's shade; preconfigured by default
        line_obs - (dict) matplotlib kwargs for formatting observed value marker; preconfigured by default
        bandwidth - (str or float) density plot's bandwidth parameter or seaborn selection method therefor; 'scott' by default
        xlabel - (str) label of X-axis; 'Year' by default
        ylabel - (str) label of Y-axis; 'Probability density'
        title - (str) title of the figure; None by default
        image_format - (str) file format of figure file; 'pdf' by default
        output - (str) 'disk' (save output on disk; default) or 'display' (display figure to user)
        verbose - (bool) print messages to user; True by default

    Output: None
    
    """
    # Protect source data from side effects
    observed = data.copy()

    # Configure choice of year
    year_pos = year_pos if year_pos else -1
    year = observed.index[year_pos]
    # Record observed value
    observation = observed[target].iloc[year_pos]
    # Carve out relevant sections of the trace
    sigma_trace = tracedf[sigma_varname]
    tracedf.drop(sigma_varname, axis=1, inplace=True)
    tracedf = tracedf.iloc[:, year_pos]
    # Carve out a small sample of the trace
    chainlen = tracedf.shape[0]
    samplesize = chainlen if (samplesize == 'full') else samplesize
    selection = np.random.choice(np.arange(chainlen), size=samplesize, replace=False)
    tracedf, sigma = tracedf.iloc[selection], sigma_trace[selection]
    # Convert from posterior parameter trace to posterior predictive trace
    noise = np.random.normal(0, sigma)
    ppc = tracedf + noise
    # Convert to treatment effects if appropriate
    if difference:
        ppc = observation - ppc
    # Draw up the baseline graph
    sns.kdeplot(ppc, bw=bandwidth, **line_kde)
    sns.kdeplot(ppc, bw=bandwidth, alpha=alpha, **shade_kde)
    # Draw observed value/0 treatment effect vertical line
    null_value = 0 if difference else observation
    plt.axvline(null_value, **line_obs)
    # Add labels, and optionally add a title
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)
    # If configured so, save the graph on disk
    if output == 'disk':
        # Configure file paths
        filename = filename if filename else 'ppc_bell_' + str(year) + '.' + image_format
        if not dstdir:
            cwd = Path(__file__).parent.resolve()
            dstdir = os.path.join(cwd, 'BSCFigures', 'L' + str(m) + target)
        # Create directory path if necessary
        if not os.path.exists(dstdir):
            os.makedirs(dstdir)
        # Save the figure
        dst = os.path.join(dstdir, filename)
        plt.savefig(dst, bbox_inches='tight')
        plt.close()
        # Report to user
        if verbose:
            msg = 'Figure saved: {}'.format(dst)
            print(msg)
    # Otherwise simply display the figure to user
    elif output == 'display':
        plt.show()
    else:
        msg = "Option 'output={}' not supported".format(output)
        raise NotImplementedError(msg)
    return None

