package gostruct

type MStruct struct {
	innerMap map[int]int
	flagMap  map[int]int

	setAll bool
	setVal int
}

func NewMStruct() *MStruct {
	return &MStruct{
		innerMap: make(map[int]int),
		flagMap:  make(map[int]int),
	}
}

func (m *MStruct) Get(key int) (int, bool) {
	if _, ok := m.innerMap[key]; !ok {
		return 0, false
	}

	if v, ok := m.flagMap[key]; ok {
		return v, ok
	}

	return m.setVal, true
}
func (m *MStruct) Put(key, value int) {
	m.innerMap[key] = value
	m.setAll = false
	m.flagMap[key] = value
}
func (m *MStruct) PutAll(value int) {
	m.setAll = true
	m.setVal = value
	m.flagMap = make(map[int]int)
}
