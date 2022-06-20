# Databricks notebook source
# MAGIC %md
# MAGIC 
# MAGIC ## Overview
# MAGIC 
# MAGIC This notebook will show you how to create and query a table or DataFrame that you uploaded to DBFS. [DBFS](https://docs.databricks.com/user-guide/dbfs-databricks-file-system.html) is a Databricks File System that allows you to store data for querying inside of Databricks. This notebook assumes that you have a file already inside of DBFS that you would like to read from.
# MAGIC 
# MAGIC This notebook is written in **Python** so the default cell type is Python. However, you can use different languages by using the `%LANGUAGE` syntax. Python, Scala, SQL, and R are all supported.

# COMMAND ----------

# File location and type
file_location = "/FileStore/tables/test5.csv"
file_type = "csv"

df =spark.read.csv(file_location,header=True,inferSchema=True)
df.show()

# COMMAND ----------

df.printSchema()

# COMMAND ----------

from pyspark.ml.feature import StringIndexer

# COMMAND ----------

indexer = StringIndexer(inputCol="sex",outputCol="Sex_indexed")
df_r = indexer.fit(df).transform(df)
df_r.show()

# COMMAND ----------

indexer = StringIndexer(inputCols=["smoker","day","time"],outputCols=["smoker_indexed","day_indexed","time_indexed"])
df_r = indexer.fit(df_r).transform(df_r)
df_r.show()

# COMMAND ----------

df_r.columns

# COMMAND ----------

from pyspark.ml.feature import VectorAssembler
featureassembler = VectorAssembler(inputCols=['tip','size','Sex_indexed','smoker_indexed','day_indexed','time_indexed'],outputCol="Independent Features")

output = featureassembler.transform(df_r)

# COMMAND ----------

output.show()

# COMMAND ----------

output.select('Independent Features').show()

# COMMAND ----------

finalized_data = output.select('Independent Features','total_bill')

# COMMAND ----------

finalized_data.show()

# COMMAND ----------

from pyspark.ml.regression import LinearRegression
train_data,test_data = finalized_data.randomSplit([0.75,0.25])
regressor = LinearRegression(featuresCol='Independent Features',labelCol='total_bill')
regressor = regressor.fit(train_data)

# COMMAND ----------

regressor.coefficients

# COMMAND ----------

regressor.intercept

# COMMAND ----------

pred_result = regressor.evaluate(test_data)

# COMMAND ----------

pred_result.predictions.show()

# COMMAND ----------


