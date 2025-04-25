from pyvi import ViTokenizer, ViPosTagger

ViTokenizer.tokenize(u"Trường đại học bách khoa hà nội")

ViPosTagger.postagging(ViTokenizer.tokenize(u"Trường đại học Bách Khoa Hà Nội"))

from pyvi import ViUtils
ViUtils.remove_accents(u"Trường đại học bách khoa hà nội")

from pyvi import ViUtils
ViUtils.add_accents(u'truong dai hoc bach khoa ha noi')