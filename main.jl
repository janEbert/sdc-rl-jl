import Dates
import Logging
import Random

import Flux
import JLD2
import Zygote

include("sdc_env.jl")
include("CFlux.jl")

# Arguments

const seed = 0
const test_on_random = false
const use_fixed_test_rng = false
const test_seed = 1

const M = 3
const dt = 1.0
const restol = 1e-10
const max_sequence_length = 4
const max_episode_length = 50
const lambda_real_interval = (-100, 0)
const lambda_imag_interval = (0, 0)

const lr = 0.0003
const hidden_layers = [64, 64]
const hidden_layer_type = Flux.Dense
# const hidden_layer_type = LSTM
# const activation_function = identity
# const activation_function = sigmoid
# const activation_function = tanh
const activation_function = crelu
# const activation_function = Flux.relu
# const activation_function = softplus
# const activation_function = silu
const predict_stepwise = true

const use_complex_numbers = true
# See at the bottom of this section, list item 2:
# https://jax.readthedocs.io/en/latest/notebooks/autodiff_cookbook.html#complex-numbers-and-differentiation
# "We can use `grad` to optimize f: ℂ → ℝ functions, like real-valued
# loss functions of complex parameters `x`, by taking steps in the
# [direction] of the conjugate of `grad(f)(x)`."
#
# In one test, conjugating was a bit better at some point (but
# converged towards the same, worse loss), at another, _not_
# conjugating was _much_ better.
const conjugate_gradients = false
const convert_input_to_real = false
const convert_output_to_real = false

const num_episodes = 50000
const test_interval = 250
const num_training_test_episodes = 20
const num_test_episodes = 5000

# Calculated from arguments

const PARAMETER_TYPE = use_complex_numbers ? Complex{Float64} : Float64
const INPUT_SIZE = (M * 2 + 1, max_sequence_length)
const OUTPUT_SIZE = M
const PADDING_ZEROS = zeros(PARAMETER_TYPE, INPUT_SIZE)

function build_layer(input_size, output_size, activation, initW, initb,
                     hidden_layer_type)
    layer = hidden_layer_type(input_size, output_size, activation,
                              initW=initW, initb=initb)
    layer
end

function build_layer(input_size, output_size, ::Any, initW, initb,
                     hidden_layer_type::typeof(LSTM))
    layer = hidden_layer_type(input_size, output_size, initW=initW, initb=initb,
                              init_state=(dims...) -> zeros(PARAMETER_TYPE, dims...))
    layer
end

function build_layer(input_size, output_size, activation; initW, initb)
    build_layer(input_size, output_size, activation, initW, initb, hidden_layer_type)
end

function build_model(hidden_layers)
    layers = Any[x -> reshape(x, :)]
    if convert_input_to_real
        push!(layers, real)
    end
    push!(layers, build_layer(prod(INPUT_SIZE), first(hidden_layers),
                              activation_function,
                              initW=glorot_uniform(PARAMETER_TYPE),
                              initb=init_bias(PARAMETER_TYPE)))
    prev_layer_size = first(hidden_layers)

    for layer_size in hidden_layers[begin + 1:end]
        layer = build_layer(prev_layer_size, layer_size, activation_function,
                            initW=glorot_uniform(PARAMETER_TYPE),
                            initb=init_bias(PARAMETER_TYPE))
        prev_layer_size = layer_size
        push!(layers, layer)
    end

    push!(layers, Flux.Dense(prev_layer_size, OUTPUT_SIZE,
                             initW=glorot_uniform(PARAMETER_TYPE),
                             initb=init_bias(PARAMETER_TYPE)))
    if convert_output_to_real
        push!(layers, real)
    end
    model = Flux.Chain(layers...)
    # Flux.gpu(model)
    model
end

function build_input(input, u, residual, num_steps)
    new_input = vcat(u, residual, num_steps)
    input = if !isnothing(input)
        hcat(input, new_input)
    else
        new_input
    end
    padded_input = hcat(input,
                        view(PADDING_ZEROS, :, 1:(max_sequence_length - size(input, 2))))
    (padded_input, input)
end

function episode_over(env)
    env.last_residual_norm < env.restol || env.num_steps >= max_episode_length
end

function step_loss(env, model, input, u, residual, action=nothing)
    if isnothing(action)
        (padded_input, input) = build_input(input, u, residual, env.num_steps)
        action = model(padded_input)
    end
    (u, residual) = step!(env, u, residual, action)

    if any(isnan, residual)
        error("Error! Encountered NaN value.")
    end
    env.last_residual_norm + env.num_steps
end

function sequence_loss(env, model, u, residual, fixed_action=nothing)
    Flux.reset!(model)

    input = nothing
    start_num_steps = env.num_steps
    total_loss = 0.0
    while !episode_over(env) && env.num_steps - start_num_steps < max_sequence_length
        loss = step_loss(env, model, input, u, residual, fixed_action)
        total_loss += loss
    end

    return total_loss
end

function value_and_gradient(f, args...)
    (value, back) = Zygote.pullback(f, args...)
    return (value, back(Zygote.sensitivity(value)))
end

function conjugate_grads(grads::Zygote.Grads)
    Zygote.Grads(conjugate_grads(grads.grads), grads.params)
end

function conjugate_grads(grads::AbstractDict)
    new_grads = empty(grads)
    for (key, value) in grads
        new_grads[key] = conjugate_grads(value)
    end
    new_grads
end

function conjugate_grads(grads::NamedTuple)
    new_tuple = Dict{Symbol, Any}()
    for key in propertynames(grads)
        value = grads[key]
        new_tuple[key] = conjugate_grads(value)
    end
    (; new_tuple...)
end

function conjugate_grads(grads::Number)
    conj(grads)
end

function conjugate_grads(grads::Nothing)
    nothing
end

function conjugate_grads(grads)
    conjugate_grads.(grads)
end

function maybe_conjugate_grads(grads)
    if !conjugate_gradients
        return grads
    end
    conjugate_grads(grads)
end

function train_episode!(env, model, opt, stepwise=true)
    (u, residual) = reset!(env)
    losses = Float64[]

    if !stepwise
        (padded_input, _) = build_input(nothing, u, residual, env.num_steps)
        action = model(padded_input)
    else
        action = nothing
    end

    while !episode_over(env)
        loss, grads = value_and_gradient(
            () -> sequence_loss(env, model, u, residual, action), Flux.params(model))
        push!(losses, loss)
        Flux.update!(opt, Flux.params(model), maybe_conjugate_grads(grads))
    end
    return losses
end

function episode_loss!(env, model, stepwise=true)
    (u, residual) = (env.initial_u, env.initial_residual)
    total_loss = 0.0

    if !stepwise
        (padded_input, _) = build_input(nothing, u, residual, env.num_steps)
        action = model(padded_input)
    else
        action = nothing
    end

    while !episode_over(env)
        total_loss += sequence_loss(env, model, u, residual, action)
    end
    total_loss
end

function test_model_random!(env, model, num_test_episodes)
    if use_fixed_test_rng
        test_rng = Random.MersenneTwister(test_seed)
    else
        test_rng = Random.GLOBAL_RNG
    end

    losses = Float64[]
    for _ in 1:num_test_episodes
        reset!(env, test_rng)
        loss = episode_loss!(env, model, predict_stepwise)
        push!(losses, loss)
    end
    losses
end

function test_model_linrange!(env, model, num_test_episodes)
    lambdas = range(0, -100, length=num_test_episodes)

    losses = Float64[]
    for lambda in lambdas
        set_lambda!(env, lambda)
        loss = episode_loss!(env, model, predict_stepwise)
        push!(losses, loss)
    end
    losses
end

function test_model!(env, model, num_test_episodes)
    if test_on_random
        return test_model_random!(env, model, num_test_episodes)
    end
    test_model_linrange!(env, model, num_test_episodes)
end

function build_argsdict()
    args = Dict{Symbol, Any}()

    args[:seed] = seed
    args[:test_on_random] = test_on_random
    args[:use_fixed_test_rng] = use_fixed_test_rng
    args[:test_seed] = test_seed

    args[:M] = M
    args[:dt] = dt
    args[:restol] = restol
    args[:max_sequence_length] = max_sequence_length
    args[:max_episode_length] = max_episode_length
    args[:lambda_real_interval] = lambda_real_interval
    args[:lambda_imag_interval] = lambda_imag_interval

    args[:lr] = lr
    args[:hidden_layers] = hidden_layers
    args[:hidden_layer_type] =  hidden_layer_type
    args[:activation_function] = activation_function
    args[:predict_stepwise] = predict_stepwise

    args[:use_complex_numbers] = use_complex_numbers
    args[:conjugate_gradients] = conjugate_gradients
    args[:convert_input_to_real] = convert_input_to_real
    args[:convert_output_to_real] = convert_output_to_real

    args[:num_episodes] = num_episodes
    args[:test_interval] = test_interval
    args[:num_training_test_episodes] = num_training_test_episodes
    args[:num_test_episodes] = num_test_episodes

    args
end

function log_with(logger, message)
    Logging.with_logger(logger) do
        @info message
    end
    @info message
end

function main()
    # Setup

    script_start_time = replace(string(Dates.now()), ':' => '-')
    Random.seed!(seed)

    logfile = open("logs_$script_start_time.jld2", "w")
    logger = Logging.SimpleLogger(logfile)
    JLD2.@save "args_$script_start_time.jld2" {compress=true} args=build_argsdict()

    model = build_model(hidden_layers)

    opt = Flux.ADAM(lr)

    env = SDCEnv{PARAMETER_TYPE}(M, dt, restol, max_sequence_length, max_episode_length,
                                 lambda_real_interval, lambda_imag_interval)

    log_with(logger, string("Started script at ", script_start_time, "."))
    test_loss = mean(test_model!(env, model, num_training_test_episodes))
    log_with(logger, string("Mean test loss after 0 episodes of training: ", test_loss))
    flush(logfile)

    episode_losses = Float64[]
    start_time = time()
    for episode in 1:num_episodes
        losses = train_episode!(env, model, opt, predict_stepwise)
        push!(episode_losses, sum(losses))

        if episode % test_interval == 0
            test_loss = mean(test_model!(env, model, num_training_test_episodes))
            log_with(logger,
                     string(episode, " episodes; Mean test loss: ", test_loss, "; ",
                            "Duration: ", time() - start_time, " sec"))
            flush(logfile)
        end
    end
    log_with(logger, string("Done after ", time() - start_time, " seconds!"))

    JLD2.@save "episode_losses_$script_start_time.jld2" {compress=true} episode_losses
    JLD2.@save "model_$script_start_time.jld2" {compress=true} model=Flux.cpu(model)
    JLD2.@save "opt_$script_start_time.jld2" {compress=true} opt
    JLD2.@save("weights_$script_start_time.jld2", {compress=true},
               weights=Flux.params(model))

    close(logfile)
end


if abspath(PROGRAM_FILE) == @__FILE__
    main()
end