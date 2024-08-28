
![[multi-species-CA-examples.png]]

An extension of [[Cellular Automata]] such that N species (i.e. N different update functions $f$) are included on the lattice and can update the states in $L$. 

Here $L(t)$ represents the lattice with all the state values at time $t$, and $S(t)$ represents the lattice storing which species occupies each lattice point. Lattice states are just integer values within some range. The function $\mathcal{N}$ is used to denote the neighbourhood of a lattice - e.g. $\mathcal{N}(S(t))_{i,j}$ is the set of all species present in the neighbourhood of point $(i,j)$.

The multi-species cellular automata model considers N update functions $f_s$, which, for any point on the lattice, maps the sum of all values within a point's neighbourhood to a new updated value. When updating a point on the lattice, all the species already occupying space in the neighbourhood compete for occupation of the lattice site. Whichever $f_s$ maps the summed neighbourhood to the highest state actually performs the update and gets to occupy the lattice site. The lattices $L$ and $S$ are updated as follows:

$$L_{i,j}(t+1) = \max  \left( \left\{ f_s(\sum_{l\in \mathcal{N}(L(t))_{i,j}} l) \right\}_{s \in \mathcal{N}(S)_{i,j}} \right)$$
$$S_{i,j}(t+1) = \arg \max_s\left( \left\{ f_s(\sum_{l\in \mathcal{N}(L(t))_{i,j}} l ) \right\}_{s \in \mathcal{N}(S)_{i,j}} \right) $$ 
See how the system can be visualized below. Colours represent different CA species, each  
with it’s own update rule, and the states (values in $L$) are visualized by the darkness of the pixel.
![[multi-species-CA-states.png]]