def plot_ppc_diff(summary, cutoff, target,
                  dstdir=None, filename=None, m=None, alpha=0.333, ci_lo='ci_0.025', ci_hi='ci_0.975', ylim=None,
                  line_avg={'linestyle': '--', 'color': 'k', 'label': 'Mean predicted'},
                  line_obs={'linestyle': '-', 'color': 'k', 'label': 'Observed'},
                  line_cutoff={'linestyle': ':', 'color': 'k'},
                  xlabel='Year', ylabel='Value', title=None, image_format='pdf', output='disk',
                  verbose=True):
    """Visualize the target society's treatment effect trajectory.
    
    This function turns a pandas datframe of summary statistics
    for the target society's posterior predictive trajectory
    into a visualization of the treatment effect's trajectory.
    
    Positional arguments:
        summary - (pandas dataframe) df of summary statistics for the target trajectory
        cutoff - (int) the start year of treatment for target society
        target - (str) title of target society in df's columns

    Optional keyword arguments:
        dstdir - (str) destination path for plot output directory; automatically constructed by default
        filename - (str) destination path for plot output filename; automatically constructed by default 
        m - (int) number of factors; 4 by default
        alpha - (float) shading transparency of the confidence interval; 0.333 by default
        ci_lo - (str) variable/column name of the lower bound of the confidence interval; summarize_ppc()'s default by default
        ci_hi - (str) variable/column name of the upper bound of the confidence interval; summarize_ppc()'s default by default
        ylim - (numpy array) y-value bounds for the plot; automatically configured by default
        line_avg - (dict) matplotlib kwargs for formatting the posterior mean trajectory; preconfigured by default
        line_obs - (dict) matplotlib kwargs for formatting the observed trajectory; preconfigured by default
        line_cutoff - (dict) matplotlib kwargs for formatting the cutoff year marker; preconfigured by default
        xlabel - (str) label of X-axis; 'Year' by default
        ylabel - (str) label of Y-axis; 'Value' by default
        title - (str) title of the figure; None by default
        image_format - (str) file format of figure file; 'pdf' by default
        output - (str) 'disk' (save output on disk; default) or 'display' (display figure to user)
        verbose - (bool) print messages to user; True by default

    Output: None
    
    """
    # Protect summary object from side effects
    summary = summary.copy()
    # Parameterize
    end = summary.index[-1]
    years = np.arange(cutoff, end+1)
    # Draw up the baseline graph
    low, high, mean = summary[ci_lo][years], summary[ci_hi][years], summary['mean'][years]
    observed = summary['observed'][years]
    low, high, mean = observed - low, observed - high, observed - mean
    plt.fill_between(years, low, high, alpha=alpha)
    plt.plot(years, mean, **line_avg)
    plt.axhline(0, **line_cutoff)
    # Add labels and legend, and optionally add a title and adjust the vertical scale
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.legend()
    if ylim:
        plt.ylim(ylim)
    if title:
        plt.title(title)
    # If configured so, save the graph on disk
    if output == 'disk':
        # Configure file paths
        filename = filename if filename else 'ppc_difference.' + image_format
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

