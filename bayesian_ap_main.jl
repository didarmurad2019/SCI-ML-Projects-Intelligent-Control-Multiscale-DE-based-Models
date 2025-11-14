using DifferentialEquations
using Plots, Statistics, Random
using Flux: Chain, Dense, ADAM, params, gradient, update!, tanh

# Set random seed
Random.seed!(42)

# Your original AP model parameters
const SI = 1.2e-4
const KG = 100.0
const Vglu = 2.1
const Kglu = 80.0
const η = 0.03
const ky = 0.05

# Create neural network
function create_neural_net()
    return Chain(
        Dense(5, 16, tanh),
        Dense(16, 16, tanh),
        Dense(16, 4)
    )
end

# Neural ODE dynamics
function neural_ode_dynamics(du, u, p, t, neural_net)
    G, Ip, Y, Ra = u
    
    # Normalized inputs for better training
    nn_input = [G/200.0, Ip/50.0, Y/5.0, Ra/30.0, t/1440.0]
    nn_output = neural_net(nn_input)
    
    # Base physiological model
    insulin_mediated_uptake = SI * Y * G / (KG + G + 1e-6)
    glucose_utilization = Vglu * G / (Kglu + G + 1e-6) / (1.0 + η * Ip)
    
    # Combine physics with neural network corrections
    du[1] = -insulin_mediated_uptake - glucose_utilization + nn_output[1] * 15.0
    du[2] = -ky * Ip + nn_output[2] * 8.0
    du[3] = -ky * Y + nn_output[3] * 3.0
    du[4] = -ky * Ra + nn_output[4] * 12.0
end

# Generate realistic training data
function generate_training_data()
    ts = 0:10.0:1440.0  # 10-minute intervals
    n_points = length(ts)
    
    G_data = Vector{Float64}(undef, n_points)
    Ip_data = Vector{Float64}(undef, n_points)
    Y_data = Vector{Float64}(undef, n_points)
    Ra_data = Vector{Float64}(undef, n_points)
    
    # Meal parameters
    meal_times = [360, 720, 1080]  # Breakfast, Lunch, Dinner (minutes)
    meal_sizes = [60, 70, 80]      # Carbohydrate grams
    
    # Initial conditions
    G_data[1] = 100.0
    Ip_data[1] = 8.0
    Y_data[1] = 0.3
    Ra_data[1] = 0.0
    
    for i in 2:n_points
        t = ts[i]
        dt = ts[i] - ts[i-1]
        
        # Meal effects
        meal_effect = 0.0
        for (meal_time, meal_size) in zip(meal_times, meal_sizes)
            if t >= meal_time && t <= meal_time + 180
                τ = t - meal_time
                meal_effect += meal_size * 0.8 * exp(-τ/60) * (1 - exp(-τ/30))^2
            end
        end
        
        # Simple dynamics
        insulin_effect = SI * Y_data[i-1] * G_data[i-1] / (KG + G_data[i-1])
        utilization = Vglu * G_data[i-1] / (Kglu + G_data[i-1]) / (1.0 + η * Ip_data[i-1])
        
        G_data[i] = G_data[i-1] + dt * (
            -insulin_effect - utilization + meal_effect + 0.5 * randn()
        )
        
        Ip_data[i] = Ip_data[i-1] + dt * (
            -ky * Ip_data[i-1] + 0.01 + 0.05 * randn()
        )
        
        Y_data[i] = max(0.0, Y_data[i-1] + dt * (
            -ky * Y_data[i-1] + 0.002 * Ip_data[i-1] + 0.02 * randn()
        ))
        
        Ra_data[i] = max(0.0, Ra_data[i-1] + dt * (
            -ky * Ra_data[i-1] + meal_effect/2.0 + 0.1 * randn()
        ))
        
        # Physiological constraints
        G_data[i] = max(50.0, min(300.0, G_data[i]))
    end
    
    return ts, G_data, Ip_data, Y_data, Ra_data
end

# Train the neural ODE
function train_neural_ode(ts, G_data, Ip_data, Y_data, Ra_data; epochs=100)
    neural_net = create_neural_net()
    optimizer = ADAM(0.001)
    
    full_data = hcat(G_data, Ip_data, Y_data, Ra_data)
    u0 = [G_data[1], Ip_data[1], Y_data[1], Ra_data[1]]
    
    losses = Float64[]
    
    function compute_loss()
        try
            prob = ODEProblem(
                (du, u, p, t) -> neural_ode_dynamics(du, u, p, t, neural_net),
                u0, (ts[1], ts[end])
            )
            
            sol = solve(prob, Tsit5(), saveat=ts, abstol=1e-6, reltol=1e-6)
            
            if sol.retcode == :Success
                total_loss = 0.0
                for i in 1:length(ts)
                    pred = sol.u[i]
                    obs = full_data[i, :]
                    total_loss += sum((pred .- obs).^2)
                end
                return total_loss / length(ts)
            else
                return 1000.0
            end
        catch e
            return 1000.0
        end
    end
    
    # Training loop
    for epoch in 1:epochs
        loss_val = compute_loss()
        push!(losses, loss_val)
        
        grads = gradient(() -> compute_loss(), params(neural_net))
        if !isnothing(grads)
            update!(optimizer, params(neural_net), grads)
        end
        
        if epoch % 20 == 0
            println("Epoch $epoch, Loss: $(round(loss_val, digits=4))")
        end
    end
    
    return neural_net, losses
end

# Ensemble predictions for uncertainty
function ensemble_predictions(neural_net, ts, initial_conditions; n_ensemble=30)
    predictions = []
    
    for i in 1:n_ensemble
        try
            # Add small perturbations
            perturbed_u0 = initial_conditions .* (1.0 .+ 0.03 .* randn(4))
            
            prob = ODEProblem(
                (du, u, p, t) -> neural_ode_dynamics(du, u, p, t, neural_net),
                perturbed_u0, (ts[1], ts[end])
            )
            
            sol = solve(prob, Tsit5(), saveat=ts)
            
            if sol.retcode == :Success
                glucose_pred = [u[1] for u in sol.u]
                push!(predictions, glucose_pred)
            end
        catch e
            continue
        end
    end
    
    return predictions
end

# Main function
function main()
    println("Generating training data...")
    ts, G_data, Ip_data, Y_data, Ra_data = generate_training_data()
    
    println("Training Neural ODE...")
    neural_net, losses = train_neural_ode(ts, G_data, Ip_data, Y_data, Ra_data, epochs=80)
    
    println("Generating uncertainty estimates...")
    initial_conditions = [G_data[1], Ip_data[1], Y_data[1], Ra_data[1]]
    ensemble_preds = ensemble_predictions(neural_net, ts, initial_conditions)
    
    # Plot results
    plot_results(ts, G_data, ensemble_preds, losses)
    
    return neural_net, ensemble_preds
end

function plot_results(ts, G_data, ensemble_preds, losses)
    ts_hours = ts ./ 60.0
    
    p1 = plot(ts_hours, G_data, label="Observed Glucose", 
              linewidth=3, color=:blue, alpha=0.8)
    
    if !isempty(ensemble_preds)
        ensemble_matrix = hcat(ensemble_preds...)
        mean_pred = vec(mean(ensemble_matrix, dims=2))
        std_pred = vec(std(ensemble_matrix, dims=2))
        
        plot!(ts_hours, mean_pred, ribbon=1.96 .* std_pred, 
              label="95% Prediction Interval", linewidth=2, 
              color=:red, alpha=0.6, fillalpha=0.2)
    end
    
    hline!([70, 180], linestyle=:dash, color=:black, 
           label=["Hypoglycemia" "Hyperglycemia"])
    xlabel!("Time (hours)")
    ylabel!("Glucose (mg/dL)")
    title!("Bayesian Neural ODE Glucose Predictions")
    
    p2 = plot(1:length(losses), losses, label="Training Loss", 
              linewidth=2, color=:purple)
    xlabel!("Epoch")
    ylabel!("Loss")
    title!("Training Progress")
    
    plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig("bayesian_ap_results.png")
end

# Run the analysis
if abspath(PROGRAM_FILE) == @__FILE__
    println("Starting Bayesian AP Analysis...")
    neural_net, ensemble_preds = main()
    println("Analysis complete! Check bayesian_ap_results.png")
end
