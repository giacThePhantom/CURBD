r"""°°°
# Recurrent spiking neural networks
This notebook provides an example in how to:
    * Set up a recurrent-spiking neural network
    * Generate data to give to it as input
    * Train the network so that it matches the data

Sources for this notebook include:
    * [The CURBD tool](https://github.com/rajanlab/CURBD)
°°°"""
# |%%--%%| <NEH5SUEzp5|QhYxSMc5Z6>
r"""°°°
# Creating the network
°°°"""
# |%%--%%| <QhYxSMc5Z6|mantena>

import numpy as np
import matplotlib.pyplot as plt
import numpy.random as npr
import math

# |%%--%%| <mantena|L7xcJxIvJC>
r"""°°°
## Neurons
Neurons are modeled by the time evolution of their membrane potential $V$:

$$\tau\frac{dV_{i}}{dt} = -V + g\sum\limits_{j}J_{ij}\phi(V_{j}) + I_{i}$$

This differential equation is solved using the Euler method, so the evolution of the voltage from one timestep to another is:

$$V_{i}(t + \Delta t) = V_{i}(t) + \frac{-V_{i}(t) + g\sum\limits_{j}J_{ij}\phi(V_{j}(t)) + I_{i}(t)}{\tau}\Delta t$$
°°°"""
# |%%--%%| <L7xcJxIvJC|TpmXKvZIwD>

n_neurons = 300
g = 1.5 #Neuron's conductance
tau_neuron = 0.1 #Neuron's time constant

def init_neurons(n_neurons):
    return npr.normal(
            -0.7, #Minimum value
            0.7, #Maximum value
            (n_neurons, 1) #(n_neurons, n_timesteps)
            )

init_neurons(n_neurons)

# |%%--%%| <TpmXKvZIwD|jahVKoLM65>

def get_activity(V, phi):
    return phi(V)

def update_voltage(V, g, J, I, activity, tau, dt):
    return V + (-V + g * J.dot(activity) + I) * dt / tau

# |%%--%%| <jahVKoLM65|xN8JsfYrEi>
r"""°°°
## Synapses
The synapses are represented as a matrix $J$ of size $n \times n$. where $n$ is the number of neurons.
The value of $J_{ij}$ represents the strength of the connection between neuron $i$ and neuron $j$, such that:
* $J_{ij} > 0$ means that neuron $i$ excites neuron $j$.
* $J_{ij} < 0$ means that neuron $i$ inhibits neuron $j$.
* $J_{ij} = 0$ means that there is no connection between neuron $i$ and neuron $j$.

The matrix is initialized with random values distributed as a normal distribution and then normalized according to the number of neurons $n$:

$$J_{ij} \sim \frac{N(0, 1)}{\sqrt{n}}$$
°°°"""
# |%%--%%| <xN8JsfYrEi|tZpgGDHTAA>

def init_synapses(n_neurons):
    return npr.randn(n_neurons, n_neurons) / math.sqrt(n_neurons)
init_synapses(n_neurons)

# |%%--%%| <tZpgGDHTAA|DHjIt9fgCu>

figure, subplot = plt.subplots(1, figsize=(10, 5))
subplot.imshow(init_synapses(n_neurons))

# |%%--%%| <DHjIt9fgCu|N2x7xaeO4F>
r"""°°°
## Input
The input I is computed as:

$$I_{i}(t) = \eta_{i}(t) + (I_{i}(t - \Delta t) - \eta_{i}(t))e^{-\frac{\Delta t}{\tau}}$$

Where $\eta_{i}(t)$ is a white noise process with a time constant $\tau$.

This is done to ensure that the input of different neurons is not correlated.
°°°"""
# |%%--%%| <N2x7xaeO4F|Qcda8uk2iM>

tau_input = 0.1 #The time constant of the input
dt = 0.01 / 5 #The timestep of the input
scale_input = 0.01 #The scale of the input

def compute_input(n_neurons, tau_input, scale_input, dt, sim_time):
    n_timesteps = int(np.ceil(sim_time / dt)) #The number of timesteps
    eta = npr.randn(n_neurons, n_timesteps) * math.sqrt(tau_input / dt) #The noise process
    I = np.ones((n_neurons, n_timesteps)) #Initializing the input

    for i in range(1, n_timesteps): #Computing the input
        I[:, i] = eta[:, i] + (I[:, i - 1] - eta[:, i]) * np.exp(-dt / tau_input)

    return I * scale_input

I = compute_input(300, tau_input, scale_input, dt, 12)

# |%%--%%| <Qcda8uk2iM|c0YKNrAEh4>

figure, subplot = plt.subplots(1, figsize=(10, 5))

subplot.plot(I[0, :])
subplot.set_xlabel("Time steps")
subplot.set_ylabel("Input current")

# |%%--%%| <c0YKNrAEh4|FwK2GquHuk>

figure, subplot = plt.subplots(1, figsize=(10, 5))

subplot.imshow(I, aspect = 'auto')
subplot.set_xlabel("Time steps")
subplot.set_ylabel("Neurons")

# |%%--%%| <FwK2GquHuk|rkJqlKN2rh>
r"""°°°
## Activation function
The activation function is any non-linear function like:
* The sigmoid.
* The hyperbolic tangent.
* The rectified linear unit (ReLU)

In this case the hyperbolic tangent is used:

$$\phi(x) = \tanh(x) = \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}$$

With its derivative:

$$\phi'(x) = 1 - \tanh^{2}(x) = 1 - \frac{e^{x} - e^{-x}}{e^{x} + e^{-x}}^{2}$$
°°°"""
# |%%--%%| <rkJqlKN2rh|L1DW3X5I98>

phi = np.tanh

# |%%--%%| <L1DW3X5I98|fcbTSOnoRZ>

x = np.linspace(-5, 5, 1000)
y = phi(x)
figure, subplot = plt.subplots(1, figsize=(10, 5))
subplot.spines['left'].set_position('center')
subplot.spines['bottom'].set_position('center')
subplot.spines['right'].set_color('none')
subplot.spines['top'].set_color('none')
subplot.xaxis.set_ticks_position('bottom')
subplot.yaxis.set_ticks_position('left')
subplot.set_title("Activation function")
subplot.plot(x, y)

# |%%--%%| <fcbTSOnoRZ|nsVBtXprrj>

figure, subplot = plt.subplots(1, figsize=(10, 5))
subplot.spines['left'].set_position('center')
subplot.spines['bottom'].set_position('center')
subplot.spines['right'].set_color('none')
subplot.spines['top'].set_color('none')
subplot.xaxis.set_ticks_position('bottom')
subplot.yaxis.set_ticks_position('left')
subplot.set_title("Derivative of the activation function")
subplot.plot(x, 1 - y**2)

# |%%--%%| <nsVBtXprrj|vzC7GmU5JD>
r"""°°°
# Training
Training consists of updating the strength of the synapses connections $J_{ij}$ so that the activity of the neurons is as close as possible to the one found in the experimental data.
This is done by computing the error between the experimental data and the output of the network and then updating the synapses connections according to the error:

$$J_{ij}(t) = J_{ij}(t-1) + \Delta J_{ij}(t)$$

$$\Delta J_{ij}(t) = c\cdot e_{i}(t)\sum\limits_{k=1}^N P_{jk}(t)\phi_{k}(t)$$
°°°"""
# |%%--%%| <vzC7GmU5JD|MQgpY2zxrQ>
r"""°°°
## Error function
The error function is the difference between the experimental data and the output of the network:

$$e_{i}(t) = z_{i}(t) - a_{i}(t)$$
°°°"""
# |%%--%%| <MQgpY2zxrQ|OHXncPsWTH>

def get_error(simulated, experimental):
    return simulated - experimental

# |%%--%%| <OHXncPsWTH|FWsNOtQM3m>
r"""°°°
## Inverse cross-correlation matrix
$P$ is the probability that neuron $j$ will fire in the next timestep, so the inverse cross-correlation matrix of the firing rates of units in the network.

$$P_{ij} = \langle\phi_{i}\phi_{j}\rangle^{-1}$$

This is costly to compute at every timestep, so it is computed iteratively as:

$$P(t) = P(t-1) - c\cdot ||(P(t-1)\cdot\phi(t))|| = P(t-1) - c\cdot((P(t-1)\cdot\phi(t))(P(t-1)\cdot\phi(t))^T)$$
°°°"""
# |%%--%%| <FWsNOtQM3m|wwvyiUURv6>

learning_rate = 1.0 #Works best when set to be 1 to 10 times the overall amplitude of the external inputs

def init_P(n_neurons, learning_rate):
    return learning_rate * np.eye(n_neurons)

# |%%--%%| <wwvyiUURv6|qWKW82G6gz>

def get_activity_P(P, activity):
    return P.dot(activity)

# |%%--%%| <qWKW82G6gz|ERuwWeRy9a>

def update_P(P, activity_P, c):
    return P - c * activity_P.dot(activity_P.T)

# |%%--%%| <ERuwWeRy9a|TLRdRxgdiJ>
r"""°°°
## Scaling term
The scaling term $c$ scales the update of the synapses connections.
It is computed as:

$c = \frac{1}{1+\phi^{T}(t)\cdot(P(t-1)\cdot\phi(t))}$
°°°"""
# |%%--%%| <TLRdRxgdiJ|h78ov2I9Ex>

def get_c(activity_P, activity):
    aPa = activity.T.dot(activity_P).item() # The result is a singleton, item extracts the number
    return 1 / (1 + aPa)

# |%%--%%| <h78ov2I9Ex|NTAIKNIRyt>
r"""°°°
## Update of the synapses Matrix
Recall:

$$\Delta J_{ij}(t) = c\cdot e_{i}(t)\sum\limits_{k=1}^N P_{jk}(t)\phi_{k}(t)$$
°°°"""
# |%%--%%| <NTAIKNIRyt|4aDnFrdZTl>

def update_J(J, activity_P, error, c, target_neurons):
    return J[:, target_neurons] - c * np.outer(error.flatten(), activity_P.flatten())

# |%%--%%| <4aDnFrdZTl|7QIDJja5kb>
r"""°°°
## Training function
Now that all the function have been defined, they can be collected in a single function that performs the training of the network.
°°°"""
# |%%--%%| <7QIDJja5kb|by855Yt3K7>

def train(simulated, experimental, J, P, target_neurons):
    error = get_error(simulated, experimental)
    activity_P = get_activity_P(P, simulated[target_neurons,:])
    c = get_c(activity_P, simulated[target_neurons,:])
    P = update_P(P, activity_P, c)
    J[:, target_neurons.flatten()] = update_J(J, activity_P, error, c, target_neurons)
    return J, P, np.mean(error ** 2)

# |%%--%%| <by855Yt3K7|agBdu0TvNR>
r"""°°°
# The experimental data
To demonstrate the capabilities of the network the experimental data will be generated in silico as well.
The data will be generated according to a three region ground truth model.
Each of the tree regions will contain $100$ neurons and will be driven by a different type of input:

* The first region (A) will be driven by only by the recurrent inuts from the other two regions.
* The second region (B) will be driven by a network generating a Gaussian "bump" propagating agross the network:
* The final region (C) will be driven by a network generating a fixed point that is shifted then to another during the generation.
The fixed points are generated by sampling $SQ_{i}(t)$ at two differend time points and holding them at the sampled value for the duration of the fixed point.

External inputs are connected to half of the units in their respective regions with a fixed negative weight (inhibitory) for region B and positive (excitatory) for region C.
°°°"""
# |%%--%%| <agBdu0TvNR|2LNjZRuYSx>
r"""°°°
## Generating the external inputs
°°°"""
# |%%--%%| <2LNjZRuYSx|CCkp3w5YEQ>
r"""°°°
### The Gaussian bump
$$ SQ_{i}(t) = e^{-\frac{1}{2\sigma^2}\left(\frac{1-\sigma-Nt}{T}\right)^2}$$
Where $\sigma$ is the width of the bump across the population, $N$ the population size and $t$ the total simulation time.
°°°"""
# |%%--%%| <CCkp3w5YEQ|vF7CwS320T>

def compute_x_bump(sim_time, dt, n_neurons, width, cutoff_traslation):
    """Selects which neurons have a higher firing rate.
    They will be sampled from a Gaussian distrubtion
    """
    n_samples = int(math.ceil(sim_time / dt))
    t_data = np.arange(0, sim_time, dt)
    x_bump = np.zeros((n_neurons, n_samples))
    n_width = width * n_neurons #from percentage to number
    norm = 2*n_width **2
    cutoff = math.ceil(n_samples / 2) + cutoff_traslation

    for i in range(n_neurons):
        x_bump[i, :] = np.exp(- ((i - n_width - n_neurons * t_data / (t_data[-1] / 2)) **2 / norm))
        x_bump[i, cutoff:] = x_bump[i, cutoff]

    return x_bump

# |%%--%%| <vF7CwS320T|zoK9rq0cf4>
r"""°°°
### Sequence driving network
The sequence driving network is a network that generates a fixed point that is shifted during the simulation in a sequenctial manner across neurons.
°°°"""
# |%%--%%| <zoK9rq0cf4|d9NCDdW6KB>

def sequence_driving_network(x_bump, dt , lead_time):
    hbump = np.log((x_bump + 0.01)/(1-x_bump+0.01))
    hbump = hbump - np.min(hbump)
    hbump = hbump / np.max(hbump)
    newmat = np.tile(hbump[:, 1, np.newaxis], (1, math.ceil(lead_time / dt)))
    hbump = np.concatenate((newmat, hbump), axis=1)
    return hbump

# |%%--%%| <d9NCDdW6KB|UK5W1tRRRH>

x_bump = compute_x_bump(10, 0.01, 100, 0.1, -10)
test = sequence_driving_network(x_bump, 0.01, 2)
figure, subplot = plt.subplots(1)
plt.pcolormesh(np.arange(0, 12, 0.01), np.arange(0, 100), test)

# |%%--%%| <UK5W1tRRRH|gy9STCGxnN>
r"""°°°
### Fixed point driving network
The fixed point driving network gnerates a fixed point that it is kept fixed until a certain point during the simulation, where it is translated to another neuron.
°°°"""
# |%%--%%| <gy9STCGxnN|Mi4GDataJz>

def fixed_point_driver(xbump, sim_time, lead_time, dt, n_neurons, cutoff_traslation, before_shift, after_shift):
    """Generates a fixed point that is shifted during the simulation,
    cutoff_translation: where the shift happen
    before_shift where to pick the current from x_bump before the shift
    after_shift where to pick the current from x_bump after the shift
    """
    n_samples = int(math.ceil((sim_time - lead_time) / dt))
    x_fp = np.zeros(xbump.shape)
    cutoff = math.ceil(n_samples / 2) + cutoff_traslation
    for i in range(n_neurons):
        front = xbump[i, before_shift] * np.ones((1, cutoff))
        back = xbump[i, after_shift] * np.ones((1, n_samples - cutoff))
        x_fp[i, :] = np.concatenate((front, back), axis = 1)
    h_fp = np.log((x_fp + 0.01)/(1-x_fp+0.01))
    h_fp -= np.min(h_fp)
    h_fp = h_fp / np.max(h_fp)
    newmat = np.tile(h_fp[:, 1, np.newaxis], (1, math.ceil(lead_time/dt)))
    h_fp = np.concatenate((newmat, h_fp), axis=1)
    return h_fp

# |%%--%%| <Mi4GDataJz|XSIfrLRk46>

test = fixed_point_driver(x_bump, 12, 2, 0.01, 100, -10, 10, 300)
figure, subplot = plt.subplots(1)
plt.pcolormesh(np.arange(0, 12, 0.01), np.arange(0, 100), test)

# |%%--%%| <XSIfrLRk46|ns2PdFjA1O>
r"""°°°
## Connectivity
°°°"""
# |%%--%%| <ns2PdFjA1O|Qbdg5h3HF7>
r"""°°°
### Intra-region connectivity
Each of the three regions will have random inter-connections.
The strength of the connection is sampled by a normalized normal distribution:

$$J_{ij} \sim \mathcal{N}(0, \frac{1}{\sqrt{N_{in}}})$$

And then scaled by a chaos factor $g$:

$$J_{ij} = g\cdot J_{ij}$$

$g$ has to be sufficiently large so to facilitate chaotic dynamics in the network.
°°°"""
# |%%--%%| <Qbdg5h3HF7|xlXbPekh9D>

def intra_region_connectivity(n_neurons, g):
    J = npr.randn(n_neurons, n_neurons)
    J = (g / np.sqrt(n_neurons)) * J
    return J

# |%%--%%| <xlXbPekh9D|Q0M8UfFt7V>

J_a = intra_region_connectivity(100, 1.8)
J_b = intra_region_connectivity(100, 1.5)
J_c = intra_region_connectivity(100, 1.5)

figure, subplot = plt.subplots(1, 3)
subplot[0].imshow(J_a, aspect = 'auto')
subplot[0].set_title("RNN-A synaptic strenght")
subplot[1].imshow(J_b, aspect = 'auto')
subplot[1].set_title("RNN-B synaptic strenght")
subplot[2].imshow(J_c, aspect = 'auto')
subplot[2].set_title("RNN-C synaptic strenght")

# |%%--%%| <Q0M8UfFt7V|XC1z8bZenO>
r"""°°°
### Inter-region connectivity
The number of connection is controlled by the `frac_inter_reg`, which controls the percentage of connected neurons
°°°"""
# |%%--%%| <XC1z8bZenO|R9NUMkejtt>

def source_region_neurons_connected(n_neurons, connected_fraction):
    """Returns the array source_neurons of the neurons
    in the source population, where source_neurons[i] = 1 if
    neuron i is connected to the population and 0 otherwise
    """
    n_connected = int(n_neurons * connected_fraction)
    source_neurons = np.zeros((n_neurons, 1))
    random_connected_neurons = npr.permutation(n_neurons)
    source_neurons[random_connected_neurons[0:n_connected]] = 1
    return source_neurons

# |%%--%%| <R9NUMkejtt|h3uOj5FdsB>

frac_inter_reg = 0.05
connection_A_to_B = source_region_neurons_connected(100, frac_inter_reg)
connection_A_to_C = source_region_neurons_connected(100, frac_inter_reg)
connection_B_to_A = source_region_neurons_connected(100, frac_inter_reg)
connection_B_to_C = source_region_neurons_connected(100, frac_inter_reg)
connection_C_to_B = source_region_neurons_connected(100, frac_inter_reg)
connection_C_to_A = source_region_neurons_connected(100, frac_inter_reg)

frac_external_reg = 0.5

connection_sequence_to_B = source_region_neurons_connected(100, frac_external_reg)
connection_fixed_to_C = source_region_neurons_connected(100, frac_external_reg)

print("The number of connection between A and B is: ", np.sum(connection_A_to_B))
print("The number of connection between the sequence driver and B is: ", np.sum(connection_sequence_to_B))

# |%%--%%| <h3uOj5FdsB|tl4zcmyqGm>
r"""°°°
## Initial state
The initial state is a `n_neurons` vector of values sampled from a normal distribution.
°°°"""
# |%%--%%| <tl4zcmyqGm|WjuzEEzOvb>

def get_initial_state(n_neurons):
    return 2 * npr.rand(n_neurons, 1) - 1

# |%%--%%| <WjuzEEzOvb|7WRHXhPqwm>
r"""°°°
## Generating the data
To generate the data a number of parameters is necessary, so they are all collected in a dictionary.
°°°"""
# |%%--%%| <7WRHXhPqwm|teucre>

data_params = {
        'sim_time' : 12,
        'lead_time' : 2,
        'dt' : 0.01,
        'bump_width' : 0.2,
        'n_neurons_A' : 100,
        'n_neurons_B' : 100,
        'n_neurons_C' : 100,
        'g_A' : 1.8,
        'g_B' : 1.5,
        'g_C' : 1.5,
        'frac_inter_reg' : 0.05,
        'frac_ext_reg' : 0.5,
        'activation_function' : np.tanh,
        'tau' : 0.1,
        'amplitude_inter_region' : 0.02,
        'amplitude_sequential' : 1,
        'amplitude_fixed' : -1,
        'cutoff_traslation' : 100,
        'before_shift' : 10,
        'after_shift' : 300,
        }

n_datapoints = int(data_params['sim_time'] / data_params['dt'])
data = {}
connections = {}

# |%%--%%| <teucre|lVJAoZSZ3Z>
r"""°°°
### Setting up all the necessary variables
°°°"""
# |%%--%%| <lVJAoZSZ3Z|NQycjuSucJ>

x_bump = compute_x_bump(
        data_params['sim_time'] - data_params['lead_time'],
        data_params['dt'],
        data_params['n_neurons_B'],
        data_params['bump_width'],
        -data_params['cutoff_traslation'],
        )

data = {
        'A' : np.empty((data_params['n_neurons_A'], n_datapoints)),
        'B' : np.empty((data_params['n_neurons_B'], n_datapoints)),
        'C' : np.empty((data_params['n_neurons_C'], n_datapoints)),
        'sequential' : sequence_driving_network(
            x_bump,
            data_params['dt'],
            data_params['lead_time']
            ),
        'fixed' : fixed_point_driver(
            x_bump,
            data_params['sim_time'],
            data_params['lead_time'],
            data_params['dt'],
            data_params['n_neurons_C'],
            data_params['cutoff_traslation'],
            data_params['before_shift'],
            data_params['after_shift'],
            ),
        }

data['A'][:] = np.NaN
data['B'][:] = np.NaN
data['C'][:] = np.NaN

states = {
        'A' : get_initial_state(data_params['n_neurons_A']),
        'B' : get_initial_state(data_params['n_neurons_B']),
        'C' : get_initial_state(data_params['n_neurons_C']),
        }
connections['A'] = {
        'A' : {
            'J' : intra_region_connectivity(
                data_params['n_neurons_A'],
                data_params['g_A']
                ),
            },
        'B' : {
            'J' : source_region_neurons_connected(
                data_params['n_neurons_A'],
                data_params['frac_inter_reg']
                ),
            'w' : data_params['amplitude_inter_region'],
            },
        'C' : {
            'J' : source_region_neurons_connected(
                data_params['n_neurons_A'],
                data_params['frac_inter_reg']
                ),
            'w' : data_params['amplitude_inter_region'],
            },
        }
connections['B'] = {
        'A' : {
            'J' : source_region_neurons_connected(
                data_params['n_neurons_B'],
                data_params['frac_inter_reg']
                ),
            'w' : data_params['amplitude_inter_region'],
            },
        'B' : {
            'J' : intra_region_connectivity(
                data_params['n_neurons_B'],
                data_params['g_B']
                ),
            },
        'C' : {
            'region' : 'C',
            'J' : source_region_neurons_connected(
                data_params['n_neurons_B'],
                data_params['frac_inter_reg']
                ),
            'w' : data_params['amplitude_inter_region'],
            },
        'sequential' : {
            'J' : source_region_neurons_connected(
                data_params['n_neurons_B'],
                data_params['frac_ext_reg']
                ),
            'w' : data_params['amplitude_sequential'],
            },
        }
connections['C'] = {
        'A' : {
            'J' : source_region_neurons_connected(
                data_params['n_neurons_C'],
                data_params['frac_inter_reg']
                ),
            'w' : data_params['amplitude_inter_region'],
            },
        'B' : {
            'J' : source_region_neurons_connected(
                data_params['n_neurons_C'],
                data_params['frac_inter_reg']
                ),
            'w' : data_params['amplitude_inter_region'],
            },
        'C' : {
            'J' : intra_region_connectivity(
                data_params['n_neurons_C'],
                data_params['g_C']
                ),
        },
        'fixed' : {
            'J' : source_region_neurons_connected(
                data_params['n_neurons_C'],
                data_params['frac_ext_reg']
                ),
            'w' : data_params['amplitude_fixed'],
            },
        }

# |%%--%%| <NQycjuSucJ|rZFkp6Bip5>

figure, subplots = plt.subplots(1, 2)
subplots[0].pcolormesh(
        np.arange(0, data_params['sim_time'], data_params['dt']),
        np.arange(0, data_params['n_neurons_B']),
        data['sequential'])
subplots[0].set_title("Sequential driver inputted in B")

subplots[1].pcolormesh(
        np.arange(0, data_params['sim_time'], data_params['dt']),
        np.arange(0, data_params['n_neurons_C']),
        data['fixed'])
subplots[1].set_title("Fixed point driver inputted in C")

# |%%--%%| <rZFkp6Bip5|n7VzpFQee9>
r"""°°°
### Some helpful functions
°°°"""
# |%%--%%| <n7VzpFQee9|p2SoxuC3fL>

def update_states(data, states, connections, dt, tau, t):
    for target in states:
        JR = connections[target][target]['J'].dot(data[target][:, t, np.newaxis])
        for source in connections[target]:
            if source != target:
                JR += connections[target][source]['w'] *\
                        connections[target][source]['J'] *\
                        data[source][:, t, np.newaxis]
        states[target] += dt * (-states[target] + JR) / tau
    return states

def save_states(data, states, phi, t):
    for region in states:
        data[region][:, t, np.newaxis] = phi(states[region])
    return data

# |%%--%%| <p2SoxuC3fL|TmsjrC0XQK>

for t in range(n_datapoints):
    data = save_states(
            data,
            states,
            data_params['activation_function'],
            t
            )
    states = update_states(
            data,
            states,
            connections,
            data_params['dt'],
            data_params['tau'],
            t
            )

for i in data:
    data[i] = data[i] / np.max(data[i])

# |%%--%%| <TmsjrC0XQK|bAuUQSglLX>

figure, subplots = plt.subplots(1, 3)
figure.tight_layout()
figure.subplots_adjust(hspace = 0.4, wspace = 0.3)
subplots[0].pcolormesh(
        np.arange(0, data_params['sim_time'], data_params['dt']),
        np.arange(0, data_params['n_neurons_A']),
        data['A']
        )

subplots[1].pcolormesh(
        np.arange(0, data_params['n_neurons_A']),
        np.arange(0, data_params['n_neurons_A']),
                  connections['A']['A']['J']
                  )

for _ in range(3):
    idx = npr.randint(0, data_params['n_neurons_A'] - 1)
    subplots[2].plot(
            np.arange(0, data_params['sim_time'], data_params['dt']),
            data['A'][idx, :],
            )

# |%%--%%| <bAuUQSglLX|47CEdJCjbU>

figure, subplots = plt.subplots(1, 3)
figure.tight_layout()
figure.subplots_adjust(hspace = 0.4, wspace = 0.3)
subplots[0].pcolormesh(
        np.arange(0, data_params['sim_time'], data_params['dt']),
        np.arange(0, data_params['n_neurons_B']),
        data['B']
        )

subplots[1].pcolormesh(
        np.arange(0, data_params['n_neurons_B']),
        np.arange(0, data_params['n_neurons_B']),
                  connections['B']['B']['J']
                  )

for _ in range(3):
    idx = npr.randint(0, data_params['n_neurons_B'] - 1)
    subplots[2].plot(
            np.arange(0, data_params['sim_time'], data_params['dt']),
            data['B'][idx, :],
            )

# |%%--%%| <47CEdJCjbU|6ZexEQo9Lz>

figure, subplots = plt.subplots(1, 3)
figure.tight_layout()
figure.subplots_adjust(hspace = 0.4, wspace = 0.3)
subplots[0].pcolormesh(
        np.arange(0, data_params['sim_time'], data_params['dt']),
        np.arange(0, data_params['n_neurons_C']),
        data['C']
        )

subplots[1].pcolormesh(
        np.arange(0, data_params['n_neurons_C']),
        np.arange(0, data_params['n_neurons_C']),
                  connections['C']['C']['J']
                  )

for _ in range(3):
    idx = npr.randint(0, data_params['n_neurons_C'] - 1)
    subplots[2].plot(
            np.arange(0, data_params['sim_time'], data_params['dt']),
            data['C'][idx, :],
            )

# |%%--%%| <6ZexEQo9Lz|GyLWs2QXLb>
r"""°°°
# In silico experiment example
°°°"""
# |%%--%%| <GyLWs2QXLb|wr6u9jsZmm>
r"""°°°
## Initialization
°°°"""
# |%%--%%| <wr6u9jsZmm|J8pbXFiyZ0>

n_target_neurons = 300
learnList = npr.permutation(n_target_neurons)
target_neurons = learnList[:n_target_neurons]

sim_param = {
        'n_neurons' : 300,
        'g' : 1.2,
        'tau_neuron' : 0.1,
        'tau_input' : 0.1,
        'scale_input' : 0.01,
        'dt' : 0.002,
        'data_dt' : data_params['dt'],
        'sim_time' : data_params['sim_time'], #in seconds
        'learning_rate' : 1.0,
        'activation_function' : np.tanh,
        'target_neurons' : target_neurons,
        }

neurons = init_neurons(sim_param['n_neurons'])
J = init_synapses(sim_param['n_neurons'])
P = init_P(sim_param['n_neurons'], sim_param['learning_rate'])
H = compute_input(
        sim_param['n_neurons'],
        sim_param['tau_input'],
        sim_param['scale_input'],
        sim_param['dt'],
        sim_param['sim_time'])
experimental_data = np.concatenate((data['A'], data['B'], data['C']), axis = 0)
experimental_data = experimental_data/experimental_data.max()
experimental_data = np.minimum(experimental_data, 0.999)
experimental_data = np.maximum(experimental_data, -0.999)
t = 0

# |%%--%%| <J8pbXFiyZ0|z04MsMVEpT>
r"""°°°
## Training
°°°"""
# |%%--%%| <z04MsMVEpT|rFaUgGvzDZ>

def run(neurons, J, H, P, experimental_data, sim_param, training = True):
    timestep = 0
    chi2 = 0
    simulated_activity = np.zeros((sim_param['n_neurons'], int(sim_param['sim_time'] / sim_param['dt'])))
    n_timesteps = int(sim_param['sim_time'] / sim_param['dt'])
    while timestep < n_timesteps:
        activity = get_activity(neurons, sim_param['activation_function'])
        simulated_activity[:, timestep, np.newaxis] = activity
        neurons = update_voltage(neurons,
                                 sim_param['g'],
                                 J,
                                 H[:, timestep, np.newaxis],
                                 activity,
                                 sim_param['tau_neuron'],
                                 sim_param['dt'],
                                 )

        if training and timestep % int(sim_param['data_dt'] / sim_param['dt']) == 0:
            data_timestep = int(timestep / (sim_param['data_dt'] / sim_param['dt']))
            J, P, chi2_temp = train(
                    activity,
                    experimental_data[:, data_timestep, np.newaxis],
                    J,
                    P,
                    sim_param['target_neurons'],
                    )
            chi2 += chi2_temp
        timestep += 1
    return simulated_activity, J, P, chi2

# |%%--%%| <rFaUgGvzDZ|8KJcrkXPhD>

test, J, P, chi2 = run(neurons, J, H, P, experimental_data, sim_param, training = False)
distance = np.linalg.norm(experimental_data - test[:, [i*5 for i in range(experimental_data.shape[1])]])
pvar = 1 - (distance / (math.sqrt(300 * 1200)) * np.std(experimental_data)) ** 2
print(chi2, pvar)

# |%%--%%| <8KJcrkXPhD|i7GbUE3XAW>

figure, subplots = plt.subplots(1, 2)
subplots[0].imshow(test, aspect = 'auto')
subplots[1].imshow(experimental_data, aspect = 'auto')

# |%%--%%| <i7GbUE3XAW|54L3Vlhdwl>

i = 0
test = None
prev_test = None
while i < 600:
    test, J, P, chi2 = run(neurons, J, H, P, experimental_data, sim_param)
    distance = np.linalg.norm(experimental_data - test[:, [i*5 for i in range(experimental_data.shape[1])]])
    pvar = 1 - (distance / (math.sqrt(300 * 1200)) * np.std(experimental_data)) ** 2
    print(i, chi2, pvar)

    if i > 500 or i % 100 == 0:
        figure, subplots = plt.subplots(1, 2)
        subplots[0].imshow(test, aspect = 'auto')
        subplots[1].imshow(experimental_data, aspect = 'auto')
        plt.show()
    i += 1
