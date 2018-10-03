import numpy as np
import matplotlib.pyplot as plt
import parse_castep as pc
import sys

neutron_wavelength = 1.288 

# Read the castep cell file
cell = pc.parse_cell(sys.argv[1])
lattice = [cell["lattice a"],cell["lattice b"],cell["lattice c"]]
atoms = cell["atoms"]

# Get lattice vectors, reciprocal lattice vectors
a1, a2, a3 = np.array(lattice)
trip_prod  = np.dot(a1, np.cross(a2,a3))
b1 = 2*np.pi*np.cross(a2,a3)/trip_prod
b2 = 2*np.pi*np.cross(a3,a1)/trip_prod
b3 = 2*np.pi*np.cross(a1,a2)/trip_prod

# Get all relevant reciprocal lattice vectors
neutron_wavenumber = 2*np.pi/neutron_wavelength
g_max = 2*neutron_wavenumber
hmax =  int(np.ceil(g_max/np.linalg.norm(b1)))
kmax =  int(np.ceil(g_max/np.linalg.norm(b2)))
lmax =  int(np.ceil(g_max/np.linalg.norm(b3)))

miller_indicies = [[1,1,0],[2,0,0],[2,1,1],[2,2,0],[3,1,0]]

# Loop over relevant reciprocal lattice vectors
recip_points = []
for h in range(-hmax, hmax+1):
	for k in range(-kmax, kmax+1):
		for l in range(-lmax, lmax+1):
			
			# Build the r.l.v
			g = h*b1 + k*b2 + l*b3
			ng = np.linalg.norm(g)

			# Reject vectors which are
			# too long to diffract off
			if ng > g_max:
				continue

			# Calculate the resulting diffraction
			# angle (2*theta) in degrees
			angle = 2*np.arccos(ng/g_max)
			angle *= 180/np.pi
			angle = 180 - angle

			# Accumulate the debye factor/structure factors
			df = 0
			sf = 0
			for a in atoms:
				pos = a[1]
				sf += np.exp(-1.j * np.dot(g, pos))
				for a2 in atoms:
					pos2 = a2[1]
					bij  = np.linalg.norm(pos-pos2)
					if ng * bij != 0:
						df += np.sin(ng * bij)/(ng * bij)
					else:
						df += 1

			# Store the results
			recip_points.append([g,h,k,l,angle,df,sf])

angles = np.arange(25,100,0.1)
pattern = []
for a in angles:
	pat = 0
	pf = (1+np.cos(a*np.pi/180.0)**2)/2 # Polarization factor
	broad = 0.5 # Angle broadaning
	for rp in recip_points:
		g,h,k,l,angle,df,sf = rp
		br  = np.exp(-((a-angle)/broad)**2) # Guassian broadaning
		if np.abs(sf**2) > 0.001: # Systematic absences
 			pat += df * br * pf
	pat *= np.exp(-a/50.0)
	pattern.append(pat)

plt.plot(angles, pattern)
plt.show()
