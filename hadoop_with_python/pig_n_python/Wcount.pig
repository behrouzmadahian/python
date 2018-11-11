-- defining the input and output paths:
--%default INPUT '/home/cloudera/pig_n_python/w*.txt'
--%default OUTPUT '/home/cloudera/pig_n_python/output.txt'
--loading data:
--for some reasong there is problem loading from local! I put the data on hdfs:
wordfile= LOAD '/home/cloudera/pig_n_python/w*.txt' USING PigStorage('\n') AS (linesin:chararray);
DESCRIBE wordfile;
dump wordfile;

wordfile_flat = foreach  wordfile GENERATE FLATTEN(TOKENIZE(linesin)) AS word;
tmp = LIMIT wordfile_flat 10;
dump tmp

describe wordfile_flat;
wordfile_grouped = GROUP wordfile_flat BY word;
describe wordfile_grouped;

wordcount = foreach wordfile_grouped GENERATE group AS word, COUNT(wordfile_flat.word) AS count;
describe wordcount;
tmp = limit wordcount 20;
--dump tmp;
rs = ORDER wordcount BY count;
dump rs;
--storing data
rmf /home/cloudera/pig_n_python/OUTPUT
STORE rs INTO '/home/cloudera/pig_n_python/OUTPUT' USING PigStorage('|');

