# Data source

Data can be downloaded from [this website](https://physionet.org/content/ecgiddb/1.0.0/).  
Alternatively you can run following command to get ito
> wget -r -N -c -np https://physionet.org/files/ecgiddb/1.0.0/

Once the files are downloaded run following command to extract files into expected format
> mv physionet.org/files/ecgiddb/1.0.0/* .

And afterwards delete original folder
> rm -rf physionet.org/ 

Afterwards you can cleanup the rest with
> rm ANNOTATORS README RECORDS SHA256SUMS.txt index.html biometric.shtml 