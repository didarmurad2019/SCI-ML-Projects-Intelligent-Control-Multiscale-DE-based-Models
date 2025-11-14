# ap_sim.jl
# Simple Euler-Maruyama simulator for the six-state SDDE model (approximate)
# Based on "Automated Insulin Delivery Intelligent Model" (provided parameters)
#
# Usage: julia ap_sim.jl
using Random, Statistics, Plots

# -------------------------
# PARAMETERS (from table)
# -------------------------
const SI      = 1.2e-4         # min^-1 · (µU/mL)^-1
const KG      = 100.0          # mg/dL
const Vglu    = 2.1            # mg/dL·min^-1
const Kglu    = 80.0           # mg/dL
const η       = 0.03           # (µU/mL)^-1
const kg      = 0.025          # min^-1
const Gsat    = 300.0          # mg/dL
const EGPbase = 1.1            # mg/dL·min^-1
const αE      = 1.5
const s50     = 0.05
const n       = 2
const τi      = 10.0           # min
const τg      = 15.0           # min

const bu      = 0.8
const βu      = 0.5

const ka1_0   = 0.05
const κa1     = 0.3
const γa1     = 0.2

const ka2_0   = 0.03
const ϕa2     = 0.2
const Ka2     = 50.0

const kins    = 0.4
const δ       = 0.05

const ke0     = 0.12
const ke1     = 0.05
const Ke      = 50.0

const ky      = 0.05
const kyI     = 0.01
const ny      = 3.0
const s0      = 0.05
const ρ       = 0.1
const ω       = 2π / (24*60)   # rad/min (circadian across 24 h)

const km      = 0.8
const Rmax    = 25.0
const kreflux = 0.05
const ζ       = 0.02

const T0      = 25.0           # mean absorption time (min)
const µ       = 0.2

const ε_small = 1e-6
const σ_scale = 2.0e-3         # base diffusion scale (adjustable)

# -------------------------
# SIMULATION SETTINGS
# -------------------------
dt = 1.0                     # minute time-step
tmax = 24*60.0               # simulate 24 hours (minutes)
N = Int(floor(tmax/dt)) + 1
ts = collect(0.0:dt:tmax)

# Distributed delay kernel truncation horizon (in minutes)
kernel_horizon = 6 * T0      # integrate up to 6*T0
Nk = Int(ceil(kernel_horizon / dt))
ks = (0:Nk-1) .* dt

# Precompute alpha-kernel values hα(τ;T0) for T0 (and will modulate if needed)
function h_alpha(τ, T)
    if τ < 0
        return 0.0
    end
    return (τ / (T^2)) * exp(-τ / T)
end

hvals = [h_alpha(k, T0) for k in ks]

# -------------------------
# helper functions
# -------------------------
ka1(x2) = ka1_0 * (1.0 + κa1 * tanh(γa1 * x2))
ka2(x3) = ka2_0 * (1.0 + ϕa2 * x3 / (Ka2 + x3 + ε_small))
ke(x4)  = ke0 + ke1 * (x4 / (Ke + x4 + ε_small))
psiE(x4) = (x4^n) / (x4^n + s50^n + ε_small)

# smooth saturation sat[0, Rmax]
function sat0R(x)
    if x <= 0 return 0.0 end
    if x >= Rmax return Rmax end
    return x
end

# Positive part
plus(x) = x > 0 ? x : 0.0

# Simple infusion controller u(t): basal + proportional when glucose high (example)
function u_control(glucose)
    basal = 0.01                 # U/min basal (example)
    Kp = 0.0006                  # proportional gain (tune as needed)
    err = glucose - 120.0        # target 120 mg/dL
    extra = err > 0 ? Kp * err : 0.0
    u = basal + extra
    # clip to plausible pump limits e.g., [0, 0.1] U/min
    return clamp(u, 0.0, 0.1)
end

# Example meal intake D(t): treat meals as carbohydrate appearance pulses (g/min)
# We'll map meal grams -> "D" amplitude in same units as convolution input via scaling.
# For simplicity we use pulses at 60, 300, 720 minutes.
meals = [(60.0, 60.0), (300.0, 70.0), (720.0, 80.0)]  # (time_min, grams)
# Scale factor to convert grams to model input units (tunable)
const carb_to_D = 0.8

function D_of_t(t)
    # represent meal as 20 minute triangular pulse centered at meal time
    s = 0.0
    for (tm, g) in meals
        width = 30.0
        if abs(t - tm) <= width/2
            # triangular shape
            s += g * (1 - (2*abs(t - tm)/width))
        end
    end
    return carb_to_D * s
end

# Helper: linear interpolation of history buffer
function hist_interp(hist_times, hist_vals, query_t)
    # hist_times assumed sorted, query_t >= hist_times[1]
    if query_t <= hist_times[1]
        return hist_vals[1]
    elseif query_t >= hist_times[end]
        return hist_vals[end]
    else
        # find interval
        idx = searchsortedfirst(hist_times, query_t)
        if hist_times[idx] == query_t
            return hist_vals[idx]
        else
            t1 = hist_times[idx-1]; t2 = hist_times[idx]
            v1 = hist_vals[idx-1]; v2 = hist_vals[idx]
            return v1 + (v2 - v1) * (query_t - t1) / (t2 - t1)
        end
    end
end

# -------------------------
# INITIAL CONDITIONS & HISTORY
# -------------------------
# States: x1 = G (glucose), x2, x3, x4 = Ip, x5 = Y (remote), x6 = Ra (gut)
x1_0 = 100.0   # mg/dL
x2_0 = 0.0
x3_0 = 0.0
x4_0 = 10.0    # baseline plasma insulin µU/mL
x5_0 = 0.0
x6_0 = 0.0

# History arrays contain states for times <= 0; choose constant history = initial conditions
hist_times = collect(-ceil(Int, max(τi, τg)/dt)*dt:dt:0.0)
hist_len = length(hist_times)
hist_vals = [ (x1_0, x2_0, x3_0, x4_0, x5_0, x6_0) for _ in hist_times ]

# Prepare solution arrays
G = zeros(Float64, N)
x2 = zeros(Float64, N)
x3 = zeros(Float64, N)
Ip = zeros(Float64, N)
Y = zeros(Float64, N)
Ra = zeros(Float64, N)
U = zeros(Float64, N)

# initialize
G[1]  = x1_0
x2[1] = x2_0
x3[1] = x3_0
Ip[1] = x4_0
Y[1]  = x5_0
Ra[1] = x6_0
U[1]  = u_control(G[1])

# To interpolate history on the fly, maintain a rolling buffer of times and values
past_times = copy(hist_times)
past_vals  = copy(hist_vals)    # vector of tuples

# random seed
Random.seed!(1234)

# -------------------------
# MAIN SIMULATION LOOP
# -------------------------
for k in 2:N
    t = ts[k]
    # --- delayed arguments ---
    t_taui = t - τi
    t_taug = t - τg

    # get delayed states via hist_interp
    # construct arrays for each state from past_vals
    past_t_arr = past_times
    past_x1 = [v[1] for v in past_vals]
    past_x2 = [v[2] for v in past_vals]
    past_x3 = [v[3] for v in past_vals]
    past_x4 = [v[4] for v in past_vals]
    past_x5 = [v[5] for v in past_vals]
    past_x6 = [v[6] for v in past_vals]

    x1_taui = hist_interp(past_t_arr, past_x1, t_taui)
    x4_taui = hist_interp(past_t_arr, past_x4, t_taui)
    x6_taug = hist_interp(past_t_arr, past_x6, t_taug)

    # --- distributed delay convolution for Ra (x6) ---
    # T modulation simplified: T(t)=T0*(1+µ*fphys(t)) ; we set fphys=0 for now
    Tt = T0
    # recompute kernel for Tt if needed (we reuse hvals computed at T0)
    # approximate integral integral_0^∞ D(t-τ) hα(τ;Tt) dτ via discrete sum
    conv = 0.0
    for j in 1:length(ks)
        τ = ks[j]
        t_arg = t - τ
        if t_arg >= 0
            Dval = D_of_t(t_arg)
        else
            # before t=0, use 0
            Dval = 0.0
        end
        conv += Dval * hvals[j] * dt
    end
    Ra_in = km * conv
    Ra_sat = sat0R(Ra_in)

    # read current states (previous step)
    G_prev = G[k-1]; x2_prev = x2[k-1]; x3_prev = x3[k-1]
    Ip_prev = Ip[k-1]; Y_prev = Y[k-1]; Ra_prev = Ra[k-1]

    # insulin infusion
    u = u_control(G_prev)
    U[k] = u

    # noise strengths (simple scaling functions)
    σG = σ_scale
    σx2 = σ_scale
    σx3 = σ_scale
    σI  = σ_scale
    σy  = σ_scale
    σRa = σ_scale

    # compute deterministic drifts (from model)
    # dx1/dt
    uptake_insulin = SI * Y_prev * G_prev / (KG + G_prev + ε_small)    # note x5 is remote insulin Y(t-τi) in paper: we use delayed x5?
    # In paper dx1 has SI(t) * x5(t-τi) * x1(t) / (KG + x1)
    # Using x5 at previous step (we approximated delay in x1_taui but using Y_prev for simplicity)
    uptake_insulin = SI * (hist_interp(past_t_arr, past_x5, t - τi)) * G_prev / (KG + G_prev + ε_small)

    peripheral = Vglu * G_prev / (Kglu + G_prev + ε_small) * (1.0 / (1.0 + η * Ip_prev))
    gut_term = kg * x6_taug * (1.0 - G_prev / Gsat)
    EGP = EGPbase * exp(-αE * psiE(Ip_prev))

    drift_x1 = - uptake_insulin - peripheral + gut_term + EGP

    # dx2/dt
    drift_x2 = - ka1(x2_prev) * x2_prev + bu * u / (1.0 + βu * u)

    # dx3/dt
    drift_x3 = - ka2(x3_prev) * x3_prev + ka1(x2_prev) * x2_prev

    # dx4/dt (plasma insulin)
    release = kins * x3_prev / (1.0 + δ * x3_prev)
    clearance = ke(Ip_prev) * Ip_prev  # Rclear omitted
    drift_x4 = - clearance + release

    # dx5/dt (remote insulin effect) uses delayed plasma insulin
    Ip_taui = hist_interp(past_t_arr, past_x4, t - τi)
    activation = kyI * (Ip_taui^ny) / (s0^ny + Ip_taui^ny) * (1.0 + ρ * sin(ω * t))
    drift_x5 = - ky * Y_prev + activation

    # dx6/dt (Ra)
    drift_x6 = - ky * Ra_prev + Ra_sat + kreflux * Ra_prev / (1.0 + ζ * Ra_prev)

    # stochastic increments (Euler-Maruyama)
    dW_G  = sqrt(dt) * randn()  * σG * G_prev
    dW_x2 = sqrt(dt) * randn()  * σx2 * sqrt(x2_prev^2 + ε_small)
    dW_x3 = sqrt(dt) * randn()  * σx3 * sqrt(x3_prev^2 + ε_small)
    dW_x4 = sqrt(dt) * randn()  * σI  * Ip_prev
    dW_x5 = sqrt(dt) * randn()  * σy  * sqrt(abs(Y_prev)+ε_small)
    dW_x6 = sqrt(dt) * randn()  * σRa * sqrt(abs(Ra_prev)+ε_small)

    # forward Euler-Maruyama update
    G_new  = max(0.0, G_prev  + dt * drift_x1  + dW_G)
    x2_new = max(0.0, x2_prev + dt * drift_x2 + dW_x2)
    x3_new = max(0.0, x3_prev + dt * drift_x3 + dW_x3)
    Ip_new = max(0.0, Ip_prev + dt * drift_x4 + dW_x4)
    Y_new  = max(0.0, Y_prev  + dt * drift_x5 + dW_x5)
    Ra_new = max(0.0, Ra_prev + dt * drift_x6 + dW_x6)

    # store
    G[k]  = G_new
    x2[k] = x2_new
    x3[k] = x3_new
    Ip[k] = Ip_new
    Y[k]  = Y_new
    Ra[k] = Ra_new

    # append to history buffers
    push!(past_times, t)
    push!(past_vals, (G_new, x2_new, x3_new, Ip_new, Y_new, Ra_new))

    # optionally trim history older than kernel_horizon + max delay to keep arrays small
    min_time_keep = t - (kernel_horizon + max(τi, τg) + 60.0)  # safety margin 60 min
    if length(past_times) > 1000 && past_times[end] - past_times[1] > (kernel_horizon + max(τi, τg) + 100.0)
        # drop earlier points while keeping at least initial history length
        idx = findfirst(x -> x >= min_time_keep, past_times)
        if idx == nothing
            # keep all
        else
            past_times = past_times[idx:end]
            past_vals  = past_vals[idx:end]
        end
    end
end

# -------------------------
# PLOTTING
# -------------------------
p1 = plot(ts./60, G, xlabel="Time (hours)", ylabel="Glucose (mg/dL)", title="Simulated Glucose", legend=false)
hline!(p1, [70, 180], linestyle=[:dash, :dash], label=["Hypo threshold" "Hyper threshold"])
p2 = plot(ts./60, Ip, xlabel="Time (hours)", ylabel="Plasma Insulin (µU/mL)", title="Plasma Insulin", legend=false)
p3 = plot(ts./60, U, xlabel="Time (hours)", ylabel="Insulin infusion rate (U/min)", title="Infusion (u)", legend=false)
plot(p1, p2, p3, layout=(3,1), size=(900,800))
savefig("ap_simulation.png")
println("Simulation complete. Figure saved to ap_simulation.png")
