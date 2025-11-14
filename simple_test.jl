using DifferentialEquations
using Plots

function simple_test()
    println("Running simple test...")
    
    # Simple ODE without any neural networks
    function simple_glucose_ode(du, u, p, t)
        G, I = u
        du[1] = -0.01 * G * I  # Glucose decreases with insulin
        du[2] = -0.02 * I + 0.1  # Insulin dynamics
    end
    
    u0 = [150.0, 5.0]  # Initial glucose and insulin
    tspan = (0.0, 24.0 * 60.0)  # 24 hours in minutes
    prob = ODEProblem(simple_glucose_ode, u0, tspan)
    
    # Solve
    sol = solve(prob, Tsit5(), saveat=0:10:24*60)
    
    # Plot
    t_hours = sol.t ./ 60.0
    p1 = plot(t_hours, sol[1, :], label="Glucose", linewidth=2)
    xlabel!("Time (hours)")
    ylabel!("Glucose (mg/dL)")
    title!("Simple Glucose-Insulin Model")
    
    p2 = plot(t_hours, sol[2, :], label="Insulin", linewidth=2, color=:red)
    xlabel!("Time (hours)")
    ylabel!("Insulin")
    title!("Insulin Dynamics")
    
    plot(p1, p2, layout=(2,1), size=(800, 600))
    savefig("simple_test.png")
    
    println("Simple test completed successfully!")
    return sol
end

# Run test
sol = simple_test()
