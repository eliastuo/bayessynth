def plot_factors(dfs, data, target,
                 dstdir=None, filename=None, alpha=0.333, ci_lo='ci_0.025', ci_hi='ci_0.975', yscale=0,
                 line_avg={'linestyle': '-', 'color': 'k', 'label': 'Mean predicted', 'linewidth': 1.0},
                 xlabel='Year', ylabel='Value', title=None, image_format='pdf', output='disk',
                 verbose=True):
    """Visualize the latent factor trajectories.
    
    This function turns a list of pandas datframes of summary statistics
    of the latent factors' posterior distributions
    into a matplotlib visualization.

    Positional arguments:
        dfs - (list) list of df-s of summary statistics for factors
        data - (pandas dataframe) dataset originally analyzed
        target - (str) title of target society in df's columns

    Optional keyword arguments:
        dstdir - (str) destination path for plot output directory; automatically constructed by default
        filename - (str) destination path for plot output filename; automatically constructed by default 
        alpha - (float) shading transparency of the confidence interval; 0.333 by default
        ci_lo - (str) variable/column name of the lower bound of the confidence interval; summarize_ppc()'s default by default
        ci_hi - (str) variable/column name of the upper bound of the confidence interval; summarize_ppc()'s default by default
        yscale - (int) orders of magnitude for scaling up the y-variable upon plotting; 0 by default
        line_avg - (dict) matplotlib kwargs for formatting the posterior mean trajectory; preconfigured by default
        xlabel - (str) label of X-axis; 'Year' by default
        ylabel - (str) label of Y-axis; 'Value' by default
        title - (str) title of the figure; None by default
        image_format - (str) file format of figure file; 'pdf' by default
        output - (str) 'disk' (save output on disk; default) or 'display' (display figure to user)
        verbose - (bool) print messages to user; True by default

    Output: None
    
    """
    #Configure time period
    years = data.index.values
    zeros = '' if (yscale < 1) else " '" + ''.join(['0' for i in range(yscale)])
    coefficient = 10**yscale
    m = len(dfs)

    # Create and index matplotlib subfigures
    row_num, col_num = int(np.ceil(m/2)), 2
    rows, cols = range(row_num), range(col_num)
    figure, axes = plt.subplots(row_num, col_num, sharex='col')
    indexes = list(itertools.product(rows, cols))

    # Iterate over factors
    for i, index in enumerate(indexes):
        # Capture appropriate factor data and matplotlib subfigure
        df, axis = dfs[i].copy(), axes[index]
        # Adjust the y-scale of the graphs
        df /= coefficient
        # Visualize the posterior trajectory
        low, high, mean = df[ci_lo], df[ci_hi], df['mean']
        axis.fill_between(years, low, high, color='tab:blue', alpha=alpha)
        axis.plot(years, mean, **line_avg)
        axis.set_title('Factor #' + str(i))
        axis.set_ylabel(ylabel + zeros)

    # Top-level visual layout
    figure.tight_layout()
    # If configured so, save the graph on disk
    if output == 'disk':
        # Configure file paths
        filename = filename if filename else 'factors.' + image_format
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

