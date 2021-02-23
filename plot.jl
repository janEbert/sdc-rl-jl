import JLD2
import Plots

function main(data_filename)
    JLD2.@load data_filename episode_losses
    x = 1:length(episode_losses)

    Plots.plot(x, episode_losses, title="Loss per training episode")
    Plots.ylims!((0, 5000))
    Plots.savefig("$data_filename.pdf")
    Plots.gui()
end

if abspath(PROGRAM_FILE) == @__FILE__
    if isempty(ARGS)
        error("Please call interactively with the file of losses to plot.")
    end
    main(first(ARGS))
end
