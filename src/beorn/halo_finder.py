import numpy as np
from scipy.spatial import cKDTree
from collections import deque
from time import time

class FoFHaloFinder:
    def __init__(self, positions, linking_length, min_num_particles=10):
        """
        Initialize the FoF halo finder.

        Parameters:
        - positions: Array of particle positions (N x 3)
        - linking_length: Linking length for FoF algorithm
        - min_num_particles: Minimum number of particles for a halo
        """
        self.positions = np.array(positions)
        self.linking_length = linking_length
        self.num_particles = len(positions)
        self.min_num_particles = min_num_particles
        self.labels = np.full(self.num_particles, -1, dtype=int)  # Use NumPy array for labels

    def find_labels(self):
        """
        Perform the FoF halo finding algorithm to label all the positions.

        Returns:
        - labels: Array of halo labels for each particle
        """
        current_label = 0
        tstart = time()
        print(f'Labelling points...')
        for i in range(self.num_particles):
            if self.labels[i] != -1:
                continue  # Skip particles already assigned to a halo
            self._expand_group_iterative(i, current_label)
            current_label += 1
        print(f'...done in {time()-tstart:.1f} s')
        return self.labels
    
    def find_halos(self):
        """
        Perform the FoF halo finding algorithm.

        Returns:
        - labels: Array of halo labels for each particle
        """
        labels = self.find_labels()
        
        # Post-process: set labels to -1 for halos with fewer particles than min_num_particles
        tstart = time()
        print(f'Setting labels to -1 for halos with fewer particles than {self.min_num_particles}...')
        halo_sizes = np.bincount(self.labels[self.labels != -1])
        small_halos = np.where(halo_sizes < self.min_num_particles)[0]
        for small_halo in small_halos:
            labels[labels == small_halo] = -1
        print(f'...done in {time()-tstart:.1f} s')

        self.halo_labels = labels
        halo_properties = self.get_halo_properties()

        return halo_properties
    
    def get_halo_properties(self):
        tstart = time()
        labels = self.halo_labels
        print('Calculating properties for each halo...')
        properties = ['label','halo_size','center_of_mass','distance_averaged_size']
        halo_properties = {pp: [] for pp in properties}
        for label in np.unique(labels[labels != -1]):
            indices = np.where(labels == label)[0]
            halo_size = len(indices)
            center_of_mass = np.mean(self.positions[indices], axis=0)
            distance_averaged_size = np.sum(1 / np.linalg.norm(self.positions[indices] - center_of_mass, axis=1))

            halo_properties['label'].append(label)
            halo_properties['halo_size'].append(halo_size),
            halo_properties['center_of_mass'].append(center_of_mass),
            halo_properties['distance_averaged_size'].append(distance_averaged_size)
        print(f'...done in {time()-tstart:.1f} s')
        for ke in halo_properties.keys():
            halo_properties[ke] = np.array(halo_properties[ke])
        return halo_properties
    
    def _expand_group_recursive(self, particle_index, label):
        """
        Recursively expand the connected group of particles.

        Parameters:
        - particle_index: Index of the current particle
        - label: Label to assign to the connected group
        """
        self.labels[particle_index] = label

        # Use cKDTree for efficient nearest neighbor search
        tree = cKDTree(self.positions)
        neighbors = tree.query_ball_point(self.positions[particle_index], self.linking_length)

        for neighbor_index in neighbors:
            if self.labels[neighbor_index] == -1:
                self._expand_group_recursive(neighbor_index, label)
    
    def _expand_group_iterative(self, seed_index, label):
        """
        Iteratively expand the connected group of particles.

        Parameters:
        - seed_index: Index of the seed particle
        - label: Label to assign to the connected group
        """
        stack = deque([(seed_index, label)])

        while stack:
            particle_index, current_label = stack.pop()

            if self.labels[particle_index] == -1:
                self.labels[particle_index] = current_label

                # Use cKDTree for efficient nearest neighbor search
                tree = cKDTree(self.positions)
                neighbors = tree.query_ball_point(self.positions[particle_index], self.linking_length)

                for neighbor_index in neighbors:
                    if self.labels[neighbor_index] == -1:
                        stack.append((neighbor_index, current_label))

class FoFHaloFinderMPI(FoFHaloFinder):
    def __init__(self, positions, linking_length, min_num_particles=10):
        """
        Initialize the FoF halo finder with MPI support.

        Parameters:
        - positions: Array of particle positions (N x 3)
        - linking_length: Linking length for FoF algorithm
        - min_num_particles: Minimum number of particles for a halo
        """
        super().__init__(positions, linking_length, min_num_particles)
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

    def find_labels_parallel(self):
        """
        Perform the FoF halo finding algorithm in parallel to label all the positions.

        Returns:
        - labels: Array of halo labels for each particle
        """
        current_label = 0
        local_num_particles = self.num_particles // self.size
        local_start = self.rank * local_num_particles
        local_end = (self.rank + 1) * local_num_particles

        local_positions = self.positions[local_start:local_end]

        print(f'Process {self.rank}: Labelling points on {self.size} processors...')
        for i in range(local_num_particles):
            global_index = local_start + i
            if self.labels[global_index] != -1:
                continue  # Skip particles already assigned to a halo
            self._expand_group_iterative(global_index, current_label)
            current_label += 1

        # Gather labels from all processes
        all_labels = self.comm.gather(self.labels[local_start:local_end], root=0)

        if self.rank == 0:
            # Combine labels from all processes
            all_labels = np.concatenate(all_labels, axis=0)
            for i in range(self.num_particles):
                self.labels[i] = all_labels[i]

        print(f'Process {self.rank}: ...done')
        return self.labels

    def find_halos_parallel(self):
        """
        Perform the FoF halo finding algorithm in parallel.

        Returns:
        - labels: Array of halo labels for each particle
        """
        labels = self.find_labels_parallel()

        if self.rank == 0:
            # Post-process: set labels to -1 for halos with fewer particles than min_num_particles
            print(f'Setting labels to -1 for halos with fewer particles than {self.min_num_particles}...')
            halo_sizes = np.bincount(labels[labels != -1])
            small_halos = np.where(halo_sizes < self.min_num_particles)[0]
            for small_halo in small_halos:
                labels[labels == small_halo] = -1
            print('...done')

            self.halo_labels = labels
            halo_properties = self.get_halo_properties()

        return halo_properties
    

def generate_blobs(n_samples, centers=3, n_features=3, random_state=0):
    from sklearn.datasets import make_blobs
    from sklearn.preprocessing import MinMaxScaler
    positions, _ = make_blobs(n_samples=n_samples, centers=centers, n_features=n_features, random_state=random_state)
    scaler = MinMaxScaler().fit(positions)
    positions = scaler.transform(positions)
    return positions

if __name__ == '__main__':
    from time import time
    import matplotlib.pyplot as plt 

    # Generate random particle positions for demonstration
    positions = generate_blobs(10000, centers=4)
    print(f'{positions.shape} | {positions.min(axis=0)} | {positions.max(axis=0)}')

    # Create FoFHaloFinder instance
    tstart = time()
    linking_length  = 1/np.cbrt(positions.shape[0])*0.3 #0.2
    print(f'linking_length = {linking_length:.3f}')
    fof_halo_finder = FoFHaloFinderMPI( 
                                    positions, 
                                    linking_length=linking_length, 
                                    min_num_particles=50,
                                    )

    # Perform FoF halo finding
    halo_properties = fof_halo_finder.find_halos()
    # halo_properties = fof_halo_finder.find_halos_parallel()

    # Print the labels assigned to each particle
    labels = fof_halo_finder.halo_labels
    labl, count = np.unique(labels, return_counts=1)
    print(f'Labels: {[(ll,cc) for ll,cc in zip(labl,count)]}')
    print(f'No. of haloes found: {labl[labl!=-1].size}')
    print(f'Runtime: {time()-tstart:.1f} s')

    plt.scatter(positions[:1000,0], positions[:1000,1], c=labels[:1000])
    com = halo_properties['center_of_mass']
    plt.scatter(com[:,0], com[:,1], c='r', marker='x')
    plt.tight_layout()
    plt.show()
