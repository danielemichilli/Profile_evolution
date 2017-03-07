
#Clean the archives
paz -e paz -r *.ar
for ar in *.paz; do clean.py -F surgical -o ${ar}.clean $ar; done

#Set best DM
pam -e pT -pT *.clean
for ar in *.pT; do
  pdmp_out=`pdmp -g temp.ps $ar`
  pdmp_first=${pdmp_out% Correction =*}
  dm=${pdmp_first#*Best DM = }
  pam -e clean.dm -d $dm ${ar%.pT}.clean.dm
done
rm *.pT

#Scrunch the archives
pam -e FTp512 -FTp --setnbin 512 *.dm

