using POMDPs, POMDPModels, POMDPToolbox, QMDP
POMDPs.add_all()
#Pkg.update()
#POMDPs.add("generate_sor")

type HealthPOMDP <: POMDP{Float64, Symbol, Float64} #POMDP(state, action, observation)
    reward1::Float64 #reward for state 1 (lowest on health scale)
    reward2::Float64
    reward3::Float64
    reward4::Float64
    reward5::Float64
    discountFactor::Float64 #discount
    prob_accurate_observation::Float64 #the observation given is accurate
    #is missing observation function currently
end

function HealthPOMDP() #initializes the HealthPOMDP with original values
    return HealthPOMDP(-20,-10,0,10,20,0.9,0.95)
end;

health = HealthPOMDP() #calls it 

#states of POMDP
POMDPs.states(::HealthPOMDP) = [1.0,2.0,3.0,4.0,5.0];


#possible actions for POMDP. In our case, I was thinking of maintaining a list of actions
#and then having each index correspond to one specific action

listAllActions = [:eatMore, :goTakeAWalk, :sleepMore] #sample list of actions

POMDPs.actions(::HealthPOMDP) = [:eatMore, :goTakeAWalk, :sleepMore]
POMDPs.actions(health::HealthPOMDP, state::Float64) = POMDPs.actions(health)

exampleState = 1
exampleAction = :eatMore


#function POMDPS.action_index(::HealthPOMDP, a::Symbol)
function POMDPs.action_index(::HealthPOMDP, a::Symbol)
    if a==:eatMore
        return 1
    elseif a==:goTakeAWalk
        return 2
    elseif a==:sleepMore
        return 3
    end
    error("invalid TigerPOMDP action: $a")
end;
#end

function POMDPs.state_index(::HealthPOMDP, s::Float64)
    if (s >= 1.0 && s <= 5.0)
        return trunc(Int, s)
    error("invalid state")
    end
end;

#function returning observation space
#POMDPS.observations(::HealthPOMDP) = ["walked", "ate"] #just example possible observations inputted by user
POMDPs.observations(::HealthPOMDP) = [1.0,2.0,3.0,4.0,5.0];
POMDPs.observations(health::HealthPOMDP, s::Float64) = observations(health)

#from here till "OPTIONAL?" is a part that is needed for algorithms that do sampling, so it may not be needed.

type healthObservationDistribution
    p::Float64
    it::Vector{Float64}
end

type healthTransitionDistribution
    p::Float64
    it::Vector{Float64}
end 

healthTransitionDistribution() = healthTransitionDistribution(0.2, [1,2,3,4,5])
println(healthTransitionDistribution().p)
POMDPs.iterator(d::healthTransitionDistribution) = d.it

#healthObservationDistribution() = (0.9, [1,2,3,4,5])
#println(healthObservationDistribution())
#println(healthObservationDistribution().p)
#POMDPs.iterator(d::healthObservationDistribution) = d.it

#observation or transition pdf. Since our POMDP is discrete, it returns the p of a given element.
function POMDPs.pdf(d::healthTransitionDistribution, so::Float64)
    return 0.2
    #so ? (return d.p) : (return 1.0-d.p)
end;

POMDPs.rand(rng::AbstractRNG, d::healthTransitionDistribution) = 0.2

#OPTIONAL

#transition model

function POMDPs.transition(health::HealthPOMDP, s::Float64, a::Symbol)
    d = healthTransitionDistribution()
    #does nothing currently
    if a == :eatMore || a == :sleepMore
        d.p = 0.1
    elseif a == :goTakeAWalk
        d.p = 0.5
    elseif s == 5
        d.p = 1
    end
    d
end;

#reward model
function POMDPs.reward(health::HealthPOMDP, s::Float64, a::Symbol)
    #does nothing currently
    reward = 0.0
    if s == 1
        reward += health.reward1
    elseif s == 2
        reward += health.reward2
    elseif s == 3
        reward += health.reward3
    elseif s == 4
        reward += health.reward4
    elseif s == 5
        reward += health.reward5
    end
    return reward

end

function POMDPs.observation(health::HealthPOMDP, s::Float64, a::Symbol, sp::Float64) #a::Symbol
    #does nothing currently. Works with the noisiness of our observations
    possible = [1,2,3,4,5]
    probs = [0.2,0.2,0.2,0.2,0.2]
    return POMDPToolbox.SparseCat{possible, probs}
    d = healthTransitionDistribution()
    #if a == :eatMore
    d.p = 0.2
   #end
    #d
end;

POMDPs.discount(health::HealthPOMDP) = health.discountFactor
POMDPs.n_states(::HealthPOMDP) = 5
POMDPs.n_actions(::HealthPOMDP) = 3
POMDPs.n_observations(::HealthPOMDP) = 5; #TODO: needs to be updated

POMDPs.initial_state_distribution(health::HealthPOMDP) = healthTransitionDistribution(0.2, [1,2,3,4,5]);

#probability_check(health) # checks that both observation and transition functions give probs that sum to unity
#obs_prob_consistency_check(health) # checks the observation probabilities
#trans_prob_consistency_check(health) # check the transition probabilities

# initialize a solver and compute a policy
solver = QMDPSolver() # from QMDP
policy = solve(solver, health)
belief_updater = updater(policy) # the default QMDP belief updater (discrete Bayesian filter)

# run a short simulation with the QMDP policy
history = simulate(HistoryRecorder(max_steps=10), health, policy, belief_updater)

# look at what happened
for (s, b, a, o) in eachstep(history, "sbao")
    println("State was $s,")
    println("belief was $b,")
    println("action $a was taken,")
    println("and observation $o was received.\n")
end
println("Discounted reward was $(discounted_reward(history)).")

#function main()
 #   print("This is the main function where we will do everything")
  #  states = 0
   # actions = 0
   # possibleObservations = 0
   # rewards = 0

    #set up POMDP by calling 
