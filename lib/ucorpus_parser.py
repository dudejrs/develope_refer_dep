import re
import os
# from lib.neuronlp2.io.instance import DependencyInstance



log_uscorpus = False

def logger_uscorpus(log_string):
	if log_uscorpus :
		print(log_string)

	return

class UcorpusInstance : 
	def __init__(self, sent, tags, dep_tuples) :
		self.sent = sent
		self.tags = tags
		self.dep_tuples = dep_tuples

def cleanInstances(instances) :

	clean_index_list = []
	a,b,c = 0,0,0

	for i in range(len(instances)) :
		instance = instances[i]

		if instance.sent == "" :
			clean_index_list.append(i)
			a+=1
		if len(instance.tags) == 0 :
			clean_index_list.append(i)
			b+=1
		if len(instance.tags) != len(instance.dep_tuples) :
			clean_index_list.append(i)
			c+=1

	instances = [instances[i] for i in range(len(instances)) if i not in clean_index_list]
	logger_uscorpus('a: {}, b: {}, c: {}'.format(a,b,c))

	return instances

def destruct_instances (instances):
	doc = []
	tags = []
	dep_tuples_list = []

	for instance in instances :
		doc.append(instance.sent)
		tags.append(instance.tags)
		dep_tuples_list.append(instance.dep_tuples)

	return doc, tags, dep_tuples_list

def read_ucorpus_all(dir_path) :
	file_list = os.listdir(dir_path)

	doc_ = []
	tags_ = []
	dep_tuples_list_ = []

	for file in file_list :
		print(dir_path+'/'+file)
		instances = read_ucorpus(dir_path+'/'+file)	
		instances = cleanInstances(instances)
		doc, tags, dep_tuples_list = destruct_instances(instances)
		doc_.extend(doc)
		tags_.extend(tags)
		dep_tuples_list_.extend(dep_tuples_list)

	return doc_, tags_, dep_tuples_list_


def read_ucorpus(rfName) :

	rf = open(rfName,'r',encoding="utf16", errors='ignore')

	matcher = re.compile(r'^[0-9]+\s[0-9]+')
	i = 1
	count = 0

	line = rf.readline()
	doc = [line.strip()] 
	tags = []
	dep_tuples_list = []

	flags = "IN"
	instances = []
	sent= line.strip()
	dep_tuples = []

	while len(line) != 0 :

		line = rf.readline()

		if len(line) == 0 :
			break

		if flags == "IN" :
			if line[0] == "#" :
				# if(line[1] == "0") :
				# 	sent = ""
				# 	tags = []
				# 	flags="OUT"
				pass
			elif len(line.strip()) == 0 :
				instances.append(UcorpusInstance(sent,tags,dep_tuples))
				flags = "OUT"
			elif matcher.match(line) == None :
				line = re.sub('(_)+([0-9]+)','',line.strip())
				line = re.split(r'\s', line)
				
				try: 
					for token in line :
						
						word_token = ''
						pos_token = ''

						for subtoken in re.split(r'\+', token) :
								microtoken  =  re.split(r'\/', subtoken) 
								# microtoken[0] = re.sub(r'[\[\]\(\)]','',	microtoken[0])
								word_token = word_token + microtoken[0]
								pos_token = pos_token + " " + microtoken[1]

						pos_token = re.sub(" ","+",pos_token[1:])
						tags.append((word_token,pos_token))


				except Exception as e : 
					logger_uscorpus('{},{},{}'.format(rfName, subtoken, e))
					tags = []
					continue

			else : 
				try : 
					line_ = re.split(r'\s', line)
					dep_tuples.append((int(line_[0]), int(line_[1])))
				except Exception as e :
					logger_uscorpus('{},{}'.format(rfName, e))
					continue

		else :
			if len(line.strip()) == 0 :
				continue
			else :
				sent = line.strip()
				dep_tuples = []
				tags = []
				flags = "IN"

			
	return instances

def read_ucorpus_test(rfName) :
	instances = read_ucorpus(rfName)
	instances = cleanInstances(instances)

	doc, tags, dep_tuples_list = destruct_instances(instances)

	wfName = 'result'
	wf = open(wfName, 'w')
	wf_tags = open(wfName+".tags",'w')
	wf_deps = open(wfName+".deps",'w')

	for i in range(len(doc)) :

		wf.writelines([doc[i]+'\n'])

		wf_tags.writelines(['[ {} ] : '.format(i)])	
		for j in tags[i] :
			wf_tags.writelines(['{}/{}\t'.format(j[0],j[1])])
		wf_tags.writelines(['\n'])

		# print(i,len(dep_tuples_list[i]))
		try :
			wf_deps.writelines(['[ {} ] : '.format(i)])	
			for k in dep_tuples_list[i] :
				wf_deps.writelines(['{}/{}\t'.format(k[0],k[1])])
			wf_deps.writelines(['\n'])
		except Exception as e:
			pass
			# print(i,e)

	print(len(tags),len(dep_tuples_list))


if __name__ == '__main__' :
	log_uscorpus =True
	rfName = "./data/UCorpus_DP_SR/BGEO0320_srl.txt"
	read_ucorpus_test(rfName)


