from dataset import Dataset as mypd

data = mypd.from_csv('TPC1\\train.csv','Survived')


print("Shape: ", data.shape())

print("\nfeatures: ", data.features)
print("\ntypes: ", data.types)

print("\nHas labels: ", data.has_label())

print("\nlabel: ", data.label)

print("\nClasses da label:", data.get_classes())

print("\nTipo da classe \'Age\':", data.get_type('Age'))
print("\nNull values in \'Age\':", data.null_counter('Age'))

print('------------- Summary -------------')
print(data.summary())
print('-----------------------------------')
print("\nTipo da classe \'Sex\':", data.get_type('Sex'))
print("\nModa da classe \'Sex\': ", data.get_moda('Sex'))



