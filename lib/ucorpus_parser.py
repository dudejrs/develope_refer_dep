import re
import os
# from lib.neuronlp2.io.instance import DependencyInstance




def read_ucorpus_all(dir_path) :
	file_list = os.listdir(dir_path)

	doc_ = []
	tags_ = []
	dep_tuples_ = []

	for file in file_list :
		print(dir_path+'/'+file)
		doc, tags, dep_tuples = read_ucorpus(dir_path+'/'+file)	
		doc_.append(doc)
		tags_.append(tags)
		dep_tuples_.append(dep_tuples)

	return 


def read_ucorpus(rfName) :

	rf = open(rfName,'r',encoding="utf16", errors='ignore')


	count = 0 
	i = 1

	line = rf.readline()
	doc = [line.strip()] 
	tags = []
	dep_tuples_list = []

	while len(line) != 0  :

		line = rf.readline()

		if len(line) == 0 :
			break

		elif i == 0 and len(line) > 1 : 
			doc.append(line.strip())
			i+=1
			continue

		elif i == 1 : 
			line = re.sub('(_)+([0-9]+)','',line.strip())
			line = re.split(r'\s', line)
			
			tmp = []
			for token in line :
				
				word_token = ''
				pos_token = ''

				for subtoken in re.split(r'\+', token) :
					try: 
						microtoken  =  re.split(r'\/', subtoken) 
						# microtoken[0] = re.sub(r'[\[\]\(\)]','',	microtoken[0])
						word_token = word_token + microtoken[0]
						pos_token = pos_token + " "+microtoken[1]
					except Exception as e : 
						print(rfName,len(doc),token, e)

				pos_token = re.sub(" ","+",pos_token[1:])
				tmp.append((word_token,pos_token))

			tags.append(tmp)
			i+=1
			continue

		elif line == '\n':
			i = 0
			continue

		elif line[0] != '#' :

			dep_tuples = []

			while len(line.strip()) != 0 :
				line_ = re.split(r'\s', line)
				dep_tuples.append((int(line_[0]), int(line_[1])))
				line = rf.readline()
		
			dep_tuples_list.append(dep_tuples)
			
	return doc, tags, dep_tuples_list

def read_ucorpus_test(rfName):

	doc, tags, dep_tuples_list = read_ucorpus(rfName)

	wfName = 'result'
	wf = open(wfName, 'w')
	wf_tags = open(wfName+".tags",'w')
	wf_deps = open(wfName+".deps",'w')

	for i in range(len(doc)) :
		# if( len(dep_tuples_list[i]) != len(tags[i])) :
		# 	raise Exception('태그와 dependey 튜플의 수가 같지 아니함\n count : {}\n dep_tuples : {}\n tags : {}\n'.format(i, len(dep_tuples_list[i]),len(tags[i])))
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
			print(i,e)


	print("success\n count : {}".format(len(doc)))

	print(len(tags),len(dep_tuples_list))

	return doc, tags, dep_tuples_list


if __name__ == '__main__' :
	rfName = "./data/UCorpus_DP_SR/BGEO0320_srl.txt"
	doc, tags, dep_tuples_list = read_ucorpus_test(rfName)


