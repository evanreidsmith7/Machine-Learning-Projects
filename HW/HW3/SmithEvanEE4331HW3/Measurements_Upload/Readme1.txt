The frequency domain measurement of the scattering parameter, S21, of the wireless channel was carried out using
the ZVB14 Vector Network Analyzer (VNA) from Rhode and Schwartz. The measurement system consists of the VNA, low
loss RF cables, and omnidirectional antennas at the transmitter and receiver ends. The transmitter and receiver heights were
fixed at 1.5 m. A program script was written for the VNA to measure 10 consecutive sweeps: each sweep contains 601
frequency sample points with spacing of 0.167 MHz to cover a 100 MHz band centered at 2.4 GHz. The settings
provided a high time-domain resolution of 10 nsec (inverse of the bandwidth) and a time span of (0.167 MHz) = 5.99
μsec . The measurements were designed to examine the WiFi bands specifically, channels 1, 6, and 11 in the IEEE 802.11 standard.

The frequency domain channel measurements were conducted at Khalifa University campus in Sharjah, UAE. The measurements
were done in 5 different locations:
1- Lab139 (highly cluttered)
2- Corridor_rm155 (medium cluttered) [with wall from one side and windows from the
   other side].
3- Corridor_Auditoirum (medium cluttered) [with wall from both sides].
3- Main_Lobby (low cluttered) 
4- Sports_Hall (open space). 

The layout of the floor plan is shown in Floor_Plan.pdf where the red circle represents
the transmitter and the green circle represent the receiver location.

The square area was divided into uniform grids with a spacing of one wavelength
(12.5 cm) that resulted in a total number of 196 points to capture the small-scale
variations [20]. The database was measured under a stationary scenario where
there were no movements around the Tx/Rx at the time of measurements. This is 
achieved since the survey is conducted during low-activity time. This gives
a total of 1960 samples because each point has 10 measurements.

In the case of Lab139 there is one set of measurments where the furthest point between the Tx/Rx is 7.1m and then the 
receiver will moved across the uniform grid. This gives a total of 1960 points.

In the case of Corridor_rm155 there are three set of measurements where the furthest point between Tx/Rx is 4.6,7.1,9.6m 
and this gives a total of 5880 points.

In the case of Corridor_Auditoirum there are two set of measurements where the furthest point between Tx/Rx is 4.6,9.6m 
and this gives a total of 3920 points.

In the case of Main_Lobby there are three set of measurements where the furthest point between Tx/Rx is 4.6,7.1,9.6m 
and this gives a total of 5880 points.

In the case of Sport_Hall there is one set of measurments where the furthest point between the Tx/Rx is 7.1m and then the 
receiver will moved across the uniform grid. This gives a total of 1960 points.


filename format (environment_Tx/Rx(separation)) i.e Corridor_Auditorium_7.1 : The environment is a narrow corridor with walls
from both sides and the Tx/Rx maximum separtion is 7.1m and as the reciever is moved across the uniform grid it decreases.

Each file will Loc_xxxx and this will resemble the location of the receiver and each location have ten measurements. The flow of
numbers on the uniform grid is shown in Flow_of_Numbering.pdf