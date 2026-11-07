package main

import (
	"strings"
	"testing"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/backend"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/toolworker"
)

func TestOnlyLoopbackAddressesAreAccepted(t *testing.T) {
	for addr, ok := range map[string]bool{
		"127.0.0.1:8080":   true,
		"localhost:8080":   true,
		"[::1]:8080":       true,
		"127.0.0.2:8080":   true,
		":8080":            false, // Every interface.
		"0.0.0.0:8080":     false,
		"[::]:8080":        false,
		"192.168.1.5:8080": false,
		"example.com:8080": false,
		"no-port":          false,
	} {
		if err := requireLoopback(addr); (err == nil) != ok {
			t.Errorf("%q: err = %v, want accepted = %v", addr, err, ok)
		}
	}
}

func TestTheBackendAndWorkerMustAgree(t *testing.T) {
	contract := backend.Contract{Tools: []string{"search", "calculator"}, MaxCalls: 8, MaxToolOutputChars: 4000}
	info := toolworker.Info{Tools: []string{"calculator", "search"}, MaxCalls: 8, MaxToolOutputChars: 4000}
	if err := checkAgreement(contract, info); err != nil {
		t.Fatalf("order should not matter: %v", err)
	}

	cases := map[string]func(*toolworker.Info){
		"tools":  func(i *toolworker.Info) { i.Tools = []string{"calculator"} },
		"budget": func(i *toolworker.Info) { i.MaxCalls = 7 },
		"cap":    func(i *toolworker.Info) { i.MaxToolOutputChars = 1 },
	}
	for name, change := range cases {
		t.Run(name, func(t *testing.T) {
			different := info
			change(&different)
			err := checkAgreement(contract, different)
			if err == nil {
				t.Fatal("no error")
			}
			if !strings.Contains(err.Error(), "backend") {
				t.Fatalf("err = %v", err)
			}
		})
	}
}
