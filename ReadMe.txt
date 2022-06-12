Read Me:

Code by Einar Salomonsen 
status at the time of this writing (10/06/2022): UiS master student

---------------------------------------------------------------------------------------------------------------------------------------
Some of the data used (well-logs and some segy files) are from the F3 block in the North-Sea. 
The data can be found via (https://terranubis.com/datainfo/F3-Demo-2020).
---------------------------------------------------------------------------------------------------------------------------------------
The following is copied from the thesis: 
"
E. G. Salomonsen, "ML-based porosity modeling tested on synthetic and subsurface data", 2022.
(master thesis at UiS)
Supervisor(s): Nestor Cardozo (UiS), Lothar Schulte (Shlumberger)
"
---------------------------------------------------------------------------------------------------------------------------------------
For reproducing the results of this thesis go to the "scripts" folder. 
Then run the "top_cases_script.py" and "F3_script.py", they loop over lists of parameters for example, 
the combinations of well-locations. 
So, to run a specific parameter combination simply enter the parameter values into the appropriate lists.
For example, if one wants to test window size equal to 1 for the synthetic models, 
change line 16 in "top_cases_script.py" from window_list = [0, 10] to window_list = [1].

Also many of the figures in the thesis were made using "making results.ipynb". 
It also compiles the results for investigation, showing how this is done.
---------------------------------------------------------------------------------------------------------------------------------------
The data used can be described as the following. 
The cross-sections of porosity, impedance, ect. are extracted from Petrel as 2D .segy files. 
The seismic horizon data comes in the form of a point set in 3D, directly extracted from Petrel. 
The well-paths are also directly extracted from Petrel. The well-logs are copied into an excel file. 
An example of these data file are shown in the figure below.

Before the code can be ran the data must be altered using some of the scripts. 
That said, this version contains the altered and original data, making that the following workflow unnecessary, but useful to note. 
The section_horizon_coord.py script is ran using the 2D cross-section 
and horizon point sets and produces a .npy file for each horizon named after the original horizon file +TWT. 
After this one should run make_new_wells.py to make a .csv file from the excel well-logs file, 
it's named the same as the excel file + new. 
I used the upscale.py file to upscale the well-logs based on a previously made upscaled log, 
however it does not matter how the upscaling is done. The last script for editing the data is the well_paths_horizon.py file. 
It uses the horizon files and the well path to make a .csv file with the horizon locations in the well-logs, 
called the same as the well-path name + horizon loc. 
---------------------------------------------------------------------------------------------------------------------------------------
After this it's a matter of making the case in the code. 
This is done in the  manage_cases.py file, in the __init__ function. 
In the function the neccessary files can be added to an elif block similarly to the other cases shown in the other elif blocks. 
In general the horizon+TWT files, cross-section files and the geo_int object, see the other cases for a reference. 
I made the top cases script and the F3 script to automatically produce results for all parameter values and compile the results. 
This is not strictly necessary, but should be used as a reference for how to use the manage_cases.py to produce results.

