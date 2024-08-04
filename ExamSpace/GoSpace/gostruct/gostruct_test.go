package gostruct

import (
	"testing"
)

func TestMStruct(t *testing.T) {
	m := NewMStruct()

	m.Put(1, 2)
	if v, ok := m.Get(1); !ok || v != 2 {
		t.Fatalf("test put before putAll")
	}

	m.PutAll(3)
	if v, ok := m.Get(1); !ok || v != 3 {
		t.Fatalf("test putAll")
	}

	m.Put(2, 4)
	if v, ok := m.Get(2); !ok || v != 4 {
		t.Fatalf("test new put after putAll")
	}
	if v, ok := m.Get(1); !ok || v != 3 {
		t.Fatalf("test old put before putAll")
	}

	m.PutAll(5)
	if v, ok := m.Get(1); !ok || v != 5 {
		t.Fatalf("test old put after putAll")
	}

	t.Logf("test pass")
}
