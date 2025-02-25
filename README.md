
1. after the zip file has been unzipped open the extracted directory in an IDE and install uv with 'pip install uv' in the terminal
2. enter the command line 'uv run app.py'
3. a link to the relevant URL will open up - click on it, and you will be able to upload files
4. You will be able to see the formulation_ids that were not valid after upload
or if all formulations in the file were valid - no additional comments will appear
5. After successful upload of a valid file you can access the calculated values and formulation id's uploaded so far 
and also stats about each type of experiment/instrument by pressing the relevant button. 

assumptions:
* in TNS experiment type, since the name of the formulation (AKA formulation_id) is not explicitly mentioned, I took the liberty to create a counter
that will distribute a unique natural number for each formulation sequentially.
