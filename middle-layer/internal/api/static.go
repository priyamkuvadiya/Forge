package api

import (
	"errors"
	"io/fs"
	"mime"
	"net/http"
	"path"
	"strings"
)

// The page's own security policy. It loads nothing from anywhere else and can
// only talk to this server, so a script injected through a transcript (the
// policy's text is shown on the page) could not send it anywhere. React sets
// style attributes, which style-src governs, hence 'unsafe-inline' for styles
// only; scripts get no such allowance.
const contentSecurityPolicy = "default-src 'self'; script-src 'self'; style-src 'self' 'unsafe-inline'; " +
	"img-src 'self' data:; connect-src 'self'; object-src 'none'; base-uri 'none'; frame-ancestors 'none'"

// staticFiles serves the built frontend: index.html at /, and the files under it
// by name. The page routes in its URL fragment, so no other path needs to fall
// back to index.html, and an unknown one is a 404 rather than a page that then
// fails to load its data.
func staticFiles(files fs.FS) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		name := strings.TrimPrefix(path.Clean("/"+r.URL.Path), "/")
		if name == "" {
			name = "index.html"
		}
		data, err := fs.ReadFile(files, name)
		if errors.Is(err, fs.ErrNotExist) || errors.Is(err, fs.ErrInvalid) {
			http.NotFound(w, r)
			return
		}
		if err != nil {
			// A directory, say: fs.ReadFile refuses one.
			http.NotFound(w, r)
			return
		}
		header := w.Header()
		contentType := mime.TypeByExtension(path.Ext(name))
		if contentType == "" {
			contentType = http.DetectContentType(data)
		}
		header.Set("Content-Type", contentType)
		header.Set("X-Content-Type-Options", "nosniff")
		header.Set("Content-Security-Policy", contentSecurityPolicy)
		header.Set("Referrer-Policy", "no-referrer")
		// Vite names assets by their content hash, so those can be kept; the page
		// that names them cannot.
		if strings.HasPrefix(name, "assets/") {
			header.Set("Cache-Control", "public, max-age=31536000, immutable")
		} else {
			header.Set("Cache-Control", "no-cache")
		}
		if r.Method == http.MethodHead {
			return
		}
		_, _ = w.Write(data)
	})
}
