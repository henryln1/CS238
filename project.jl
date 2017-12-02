using POMDPs, POMDPModels, POMDPToolbox, QMDP
#Pkg.checkout("POMDPs")
#POMDPs.add_all()
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

listAllActions = [:eatMore, :goTakeAWalk, :sleepMore, :eatVegetables, :drinkWater, :keepItUp] #sample list of actions

POMDPs.actions(::HealthPOMDP) = [:eatMore, :goTakeAWalk, :sleepMore, :eatVegetables, :drinkWater, :keepItUp]
POMDPs.actions(health::HealthPOMDP, state::Float64) = POMDPs.actions(health)

#exampleState = 1
#exampleAction = :eatMore


#function POMDPS.action_index(::HealthPOMDP, a::Symbol)
function POMDPs.action_index(::HealthPOMDP, a::Symbol)
    if a==:eatMore
        return 1
    elseif a==:goTakeAWalk
        return 2
    elseif a==:sleepMore
        return 3
    elseif a==:eatVegetables
        return 4
    elseif a==:drinkWater
        return 5
    elseif a==:keepItUp
        return 6
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
    #p::Float64
    p::Vector{Float64}
    it::Vector{Float64}
end

type healthTransitionDistribution
    #p::Float64
    p::Vector{Float64}
    it::Vector{Float64}
end 

healthTransitionDistribution() = healthTransitionDistribution([0.2,0.2,0.2,0.2,0.2], [1.0,2.0,3.0,4.0,5.0])
#println(healthTransitionDistribution().p)
POMDPs.iterator(d::healthTransitionDistribution) = d.it
#print healthTransitionDistribution()

#healthObservationDistribution() = (0.9, [1,2,3,4,5])
#println(healthObservationDistribution())
#println(healthObservationDistribution().p)
#POMDPs.iterator(d::healthObservationDistribution) = d.it

#observation or transition pdf. Since our POMDP is discrete, it returns the p of a given element.
function POMDPs.pdf(d::healthTransitionDistribution, so::Float64)
    state = trunc(Int, so)
    return d.p[state]
    #so ? (return d.p) : (return 1.0-d.p)
end;

function getRandomState()
    a = collect(1:5)
    temp = a[rand(1:end)]
    newTemp = convert(Float64, temp)
    return newTemp
end;



#println((float)states[rand(1:end)])
POMDPs.rand(rng::AbstractRNG, d::healthTransitionDistribution) = getRandomState()

#POMDPs.rand(rng::AbstractRNG, d::healthTransitionDistribution) = (float)states[rand(1:end)]

#OPTIONAL

#transition model

function POMDPs.transition(health::HealthPOMDP, s::Float64, a::Symbol)
    d = healthTransitionDistribution()
    #println(d)
    #does nothing currently
    if a == :eatMore || a == :sleepMore
        d.p = [0.1,0.1,0.3,0.3,0.2]
    elseif a == :goTakeAWalk
        d.p = [0.1,0.1,0.3,0.3,0.2]
    elseif a == :eatVegetables || a == :drinkWater
        d.p = [0.2,0.3,0.2,0.2,0.1]
    elseif s == 5
        d.p = [0.1,0.1,0.1,0.1,0.6]
    end
    #println(d)
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


function POMDPs.observation(health::HealthPOMDP, a::Symbol, sp::Float64) #a::Symbol
    d = healthTransitionDistribution()
    #d = healthObservationDistribution([0.05,0.10,0.30,0.30,0.35],[1.0,2.0,3.0,4.0,5.0])
    #println(d)
    #println("hello")
    #does nothing currently. Works with the noisiness of our observations
    #possible = [1,2,3,4,5]
    #probs = [0.2,0.2,0.2,0.2,0.2]
    #return POMDPToolbox.SparseCat(possible, probs)
    #if a == :eatMore
    #d.p = 
    #end
    if a == :eatMore
        d.p = [0.4,0.2,0.2,0.2,0.1]
    end
    d
end;

#

POMDPs.discount(health::HealthPOMDP) = health.discountFactor
POMDPs.n_states(::HealthPOMDP) = 5
POMDPs.n_actions(::HealthPOMDP) = 6
POMDPs.n_observations(::HealthPOMDP) = 5; #TODO: needs to be updated

POMDPs.initial_state_distribution(health::HealthPOMDP) = healthTransitionDistribution([0.05,0.15,0.6,0.15,0.05], [1.0,2.0,3.0,4.0,5.0]);

#probability_check(health) # checks that both observation and transition functions give probs that sum to unity
#obs_prob_consistency_check(health) # checks the observation probabilities
#trans_prob_consistency_check(health) # check the transition probabilities

# initialize a solver and compute a policy
solver = QMDPSolver() # from QMDP
policy = solve(solver, health)
println(policy)
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

function askForObservations()


end;

function calculateStateFromObservation(o::Vector{Float64})
    #TODO


end;

function convertToText(a::Symbol)

numberCheckIns = 30

count = 0

while count <= numberCheckIns
    println("Hello")
    stateDistribution = calculateStateFromObservation()
    action = calculateBestAction(stateDistribution)
    convertToText(action)

#function main()
 #   print("This is the main function where we will do everything")
  #  states = 0
   # actions = 0
   # possibleObservations = 0
   # rewards = 0

    #set up POMDP by calling 
