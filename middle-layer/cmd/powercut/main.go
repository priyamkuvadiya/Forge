// Command powercut checks that the session store survives losing power.
//
// "write" runs the real store (database.Open + session.Store) in a loop and prints
// a line to stdout after each call returns: what the store has promised is saved.
// Run it somewhere whose power can be cut (a VM killed outright), with stdout going
// to a file outside that machine. "check" then reopens the database and requires
// it to be intact and to hold everything that was promised.
package main

import (
	"bufio"
	"fmt"
	"os"
	"strings"

	"github.com/priyamkuvadiya/Forge/middle-layer/internal/agent"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/database"
	"github.com/priyamkuvadiya/Forge/middle-layer/internal/session"
)

func main() {
	if len(os.Args) < 3 {
		fmt.Fprintln(os.Stderr, "usage: powercut write|check DB [ACKS]")
		os.Exit(2)
	}
	db, err := database.Open(os.Args[2])
	if err != nil {
		fmt.Fprintln(os.Stderr, err)
		os.Exit(1)
	}
	store := session.NewStore(db, 1<<30)
	switch os.Args[1] {
	case "write":
		if os.Getenv("POWERCUT_UNSAFE") == "1" {
			// The control: without syncing, a real power cut must lose something,
			// or this harness is not cutting power.
			if _, err := db.Exec("PRAGMA synchronous=OFF"); err != nil {
				panic(err)
			}
		}
		write(store)
	case "check":
		var ok string
		if err := db.QueryRow("PRAGMA integrity_check").Scan(&ok); err != nil || ok != "ok" {
			fmt.Println("CORRUPT:", ok, err)
			os.Exit(1)
		}
		os.Exit(check(store, os.Args[3]))
	}
}

func write(store *session.Store) {
	filler := strings.Repeat("the quick brown fox jumps over the lazy dog ", 50)
	for i := 0; ; i++ {
		s, err := store.Create("p")
		if err != nil {
			panic(err)
		}
		fmt.Printf("created %s\n", s.ID)
		prompt := fmt.Sprintf("task %d %s", i, filler)
		begun, err := store.Begin(s.ID, prompt)
		if err != nil {
			panic(err)
		}
		fmt.Printf("begun %s %s\n", s.ID, begun.EpisodeID)
		events := []agent.Event{{Type: agent.EventStarted, Episode: begun.EpisodeID, Prompt: prompt}}
		if err := store.Progress(s.ID, events); err != nil {
			panic(err)
		}
		fmt.Printf("progress %s %s\n", s.ID, begun.EpisodeID)
		answer := fmt.Sprint(i)
		if err := store.End(s.ID, agent.Episode{
			ID: begun.EpisodeID, Policy: "p", FinishReason: agent.FinishAnswer, Answer: &answer,
			Messages: []agent.Message{{Role: "user", Content: prompt}, {Role: "assistant", Content: "<answer>" + answer + "</answer>"}},
		}); err != nil {
			panic(err)
		}
		fmt.Printf("ended %s %s %s\n", s.ID, begun.EpisodeID, answer)
	}
}

func check(store *session.Store, acks string) int {
	file, err := os.Open(acks)
	if err != nil {
		fmt.Println(err)
		return 1
	}
	defer file.Close()
	counts := map[string]int{}
	failures := 0
	fail := func(format string, args ...any) {
		failures++
		if failures <= 20 {
			fmt.Printf("LOST: "+format+"\n", args...)
		}
	}
	scanner := bufio.NewScanner(file)
	for scanner.Scan() {
		f := strings.Fields(scanner.Text())
		if len(f) < 2 {
			continue // a line cut off by the power cut
		}
		kind := f[0]
		s, err := store.Get(f[1])
		if err != nil {
			fail("%s: session %s: %v", kind, f[1], err)
			continue
		}
		counts[kind]++
		if kind == "created" {
			continue
		}
		if len(f) < 3 {
			continue
		}
		var found *agent.Episode
		for i := range s.Episodes {
			if s.Episodes[i].ID == f[2] {
				found = &s.Episodes[i]
			}
		}
		switch {
		case found == nil:
			fail("%s: episode %s missing", kind, f[2])
		case kind == "ended" && len(f) == 4 && (found.FinishReason != agent.FinishAnswer || found.Answer == nil || *found.Answer != f[3]):
			fail("ended %s reads back as %s", f[2], found.FinishReason)
		case kind != "ended" && len(found.Messages) == 0:
			fail("%s: episode %s lost its prompt", kind, f[2])
		}
	}
	fmt.Printf("promised: %v, lost: %d\n", counts, failures)
	if failures > 0 {
		return 1
	}
	return 0
}
