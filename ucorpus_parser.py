import re


def read_file() :

	rfName = "./data/UCorpus_DP_SR/BGAD0038_srl.txt"

	rf = open(rfName,'r',encoding="utf16", errors='ignore')



	count = 0 
	i = 1

	line = rf.readline()
	doc = [line.strip()] 
	tags = []
	dep_tuples_list = []

	while len(line) != 0  and count < 100:

		line = rf.readline()

		if i == 0 and len(line) > 1 : 
			doc.append(line.strip())
			i+=1
			count +=1
			continue

		elif i == 1 : 
			line = re.sub('(_)+([0-9]+)','',line.strip())
			line = re.split(r'\s', line)
			
			tmp = []
			for token in line :
				
				word_token = ''
				pos_token = ''

				for subtoken in re.split(r'\+', token) :
					microtoken  =  re.split(r'\/', subtoken) 
					word_token = word_token + microtoken[0]
					pos_token = pos_token + " "+microtoken[1]

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
			

	for i in range(count) :
		if( len(dep_tuples_list[i]) != len(tags[i])) :
			print("error")
			return

	print("success")
	print(doc[9])
	print(tags[9])

	return doc, tags, dep_tuples_list




if __name__ == '__main__' :
	read_file()

