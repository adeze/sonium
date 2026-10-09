;;; meshlang.el --- Major mode for MeshLang (.mlg) files  -*- lexical-binding: t; -*-

;; Author: Sonium project
;; Keywords: languages, mesh, acoustics
;; Package-Requires: ((emacs "27.1"))

;;; Commentary:

;; Major mode for MeshLang, a high-level geometry/mesh specification language
;; that compiles to Gmsh .geo scripts.
;;
;; Usage — add to your init.el:
;;
;;   (add-to-list 'load-path "/path/to/sonium/crates/sonium-meshlang/editors")
;;   (require 'meshlang)
;;
;; Or with use-package + :load-path:
;;
;;   (use-package meshlang
;;     :load-path "path/to/sonium/crates/sonium-meshlang/editors"
;;     :mode "\\.mlg\\'")

;;; Code:

(defgroup meshlang nil
  "Major mode for MeshLang (.mlg) files."
  :group 'languages
  :prefix "meshlang-")

(defcustom meshlang-indent-offset 4
  "Number of spaces per indentation level."
  :type 'integer
  :safe #'integerp
  :group 'meshlang)

;; ── Syntax table ──────────────────────────────────────────────────────────────
;;
;; Comments: // line comments and /* */ block comments.
;; Note: nested block comments (supported by the lexer) are not handled here;
;; Emacs's comment syntax only supports one level of nesting.

(defvar meshlang-mode-syntax-table
  (let ((st (make-syntax-table)))
    ;; // and /* */ comments (style "b" = line comment for //)
    (modify-syntax-entry ?/ ". 124b" st)
    (modify-syntax-entry ?* ". 23"   st)
    (modify-syntax-entry ?\n "> b"   st)
    ;; Strings
    (modify-syntax-entry ?\" "\""    st)
    (modify-syntax-entry ?\\ "\\"    st)
    ;; Operators — treated as punctuation so they don't break word boundaries
    (modify-syntax-entry ?| "."  st)
    (modify-syntax-entry ?& "."  st)
    (modify-syntax-entry ?! "."  st)
    (modify-syntax-entry ?= "."  st)
    (modify-syntax-entry ?< "."  st)
    (modify-syntax-entry ?> "."  st)
    (modify-syntax-entry ?+ "."  st)
    (modify-syntax-entry ?- "."  st)
    (modify-syntax-entry ?% "."  st)
    ;; Underscore is part of identifiers
    (modify-syntax-entry ?_ "_"  st)
    st)
  "Syntax table for `meshlang-mode'.")

;; ── Font-lock faces ───────────────────────────────────────────────────────────

;; font-lock-number-face was added in Emacs 28; fall back gracefully.
(defface meshlang-number-face
  '((t :inherit (font-lock-constant-face)))
  "Face for numeric literals in MeshLang."
  :group 'meshlang)

;; ── Font-lock keyword lists ───────────────────────────────────────────────────

(defconst meshlang--keywords
  '("param" "const" "import" "from" "as"
    "fn" "pub" "geometry" "mesh" "physics" "export" "struct"
    "let" "for" "in" "if" "else"
    "group" "refine" "near" "on" "boundary_layer" "default")
  "MeshLang reserved keywords.")

(defconst meshlang--builtins
  ;; Geometric primitives
  '("Box" "Cylinder" "Sphere" "Cone" "Torus" "Ellipsoid"
    "Wedge" "Prism" "Disk" "Plane" "Extrude" "Revolve"
    ;; Transforms (common pipe targets)
    "Translate" "Rotate" "Scale" "Mirror" "Fillet" "Chamfer"
    ;; Mesh algorithms
    "Delaunay" "Frontal" "Transfinite" "Structured" "Unstructured"
    ;; Physics / BC names
    "Absorbing" "Rigid" "Impedance" "Velocity" "Pressure" "Source")
  "MeshLang built-in constructor and type names (start with uppercase).")

;; Physical units — matched only after a numeric literal to avoid clashing
;; with identifiers. Uppercase units (Hz, Pa…) are also caught by the
;; TypeIdent rule below, but we give them a distinct face here.
(defconst meshlang--units-re
  (regexp-opt
   '(;; length
     "m" "mm" "cm" "um"
     ;; time
     "s" "ms"
     ;; frequency
     "Hz" "kHz"
     ;; angle
     "deg" "rad"
     ;; pressure
     "Pa" "kPa")
   nil)                 ; no \b wrappers — anchored via surrounding pattern
  "Regexp matching a single MeshLang unit token (no word boundaries).")

;; ── Font-lock rules ───────────────────────────────────────────────────────────

(defconst meshlang-font-lock-keywords
  (let* ((kw-re    (regexp-opt meshlang--keywords  'words))
         (bi-re    (regexp-opt meshlang--builtins  'words))
         ;; Number: integer or float with optional exponent
         (num-re   "[0-9]+\\(?:\\.[0-9]*\\)?\\(?:[eE][+-]?[0-9]+\\)?")
         ;; Unit immediately following a number with optional whitespace:
         ;; group 2 captures the unit token
         (num-unit-re (concat "\\(" num-re "\\)"
                              "[[:space:]]+"
                              "\\(" meshlang--units-re "\\)"
                              "\\b")))
    `(
      ;; 1. Doc comments (///) — must come before the general comment rule
      ("///.*$" . 'font-lock-doc-face)

      ;; 2. Keywords
      (,kw-re . 'font-lock-keyword-face)

      ;; 3. Boolean literals
      (,(regexp-opt '("true" "false") 'words) . 'font-lock-constant-face)

      ;; 4. Built-in constructors / type names
      (,bi-re . 'font-lock-builtin-face)

      ;; 5. Any remaining UpperCase identifier → user-defined type
      ("\\b[[:upper:]][[:alnum:]_]*\\b" . 'font-lock-type-face)

      ;; 6. Pipe arrow |> and function-return arrow ->
      ("|>"  . 'font-lock-preprocessor-face)
      ("->"  . 'font-lock-preprocessor-face)

      ;; 7. Unit tokens following a numeric literal ("1000.0 Hz", "0.3 m")
      ;;    — subgroup 2 gets the unit face
      (,num-unit-re 2 'font-lock-preprocessor-face)

      ;; 8. Numeric literals (standalone, after unit rule so no double-match)
      (,(concat "\\b" num-re "\\b") . 'meshlang-number-face)

      ;; 9. Named declaration: fn / geometry / struct name
      (,(concat "\\b\\(?:fn\\|geometry\\|struct\\)\\b"
                "\\s-+\\([[:alpha:]_][[:alnum:]_]*\\)")
       1 'font-lock-function-name-face)

      ;; 10. let bindings
      ("\\blet\\s-+\\([[:alpha:]_][[:alnum:]_]*\\)"
       1 'font-lock-variable-name-face)

      ;; 11. param / const declarations
      ("\\b\\(?:param\\|const\\)\\s-+\\([[:alpha:]_][[:alnum:]_]*\\)"
       1 'font-lock-variable-name-face)

      ;; 12. for-loop variable
      ("\\bfor\\s-+\\([[:alpha:]_][[:alnum:]_]*\\)"
       1 'font-lock-variable-name-face)

      ;; 13. Key-value setting names inside blocks (key: value)
      ("^[[:space:]]*\\([[:alpha:]_][[:alnum:]_]*\\)[[:space:]]*:"
       1 'font-lock-constant-face)
      ))
  "Font-lock keyword specification for `meshlang-mode'.")

;; ── Indentation ───────────────────────────────────────────────────────────────

(defun meshlang-indent-line ()
  "Indent the current line for `meshlang-mode'."
  (interactive)
  (let* ((ppss   (syntax-ppss (line-beginning-position)))
         (depth  (car ppss))
         (indent (* depth meshlang-indent-offset)))
    ;; A line that starts with } closes one level
    (save-excursion
      (beginning-of-line)
      (skip-chars-forward "[:space:]")
      (when (= (char-after) ?})
        (setq indent (max 0 (- indent meshlang-indent-offset)))))
    (indent-line-to indent)))

;; ── Keymap ────────────────────────────────────────────────────────────────────

(defvar meshlang-mode-map
  (let ((map (make-sparse-keymap)))
    map)
  "Keymap for `meshlang-mode'.")

;; ── Mode definition ───────────────────────────────────────────────────────────

;;;###autoload
(define-derived-mode meshlang-mode prog-mode "MeshLang"
  "Major mode for editing MeshLang (.mlg) geometry/mesh specification files.

MeshLang is a high-level language that compiles to Gmsh .geo scripts for
use with the Sonium acoustic FEM/BEM solver.

Highlights:
  - Keywords: param, const, fn, geometry, mesh, physics, export, …
  - Constructors: Box, Cylinder, Sphere, … (UpperCase identifiers)
  - Physical units: m, mm, Hz, Pa, … (when following a numeric literal)
  - Pipe operator |>, return arrow ->
  - Doc comments (///) and block comments (/* */)

Indentation is brace-driven; customize `meshlang-indent-offset' (default 4).

\\{meshlang-mode-map}"
  :syntax-table meshlang-mode-syntax-table

  ;; Comments
  (setq-local comment-start      "// ")
  (setq-local comment-end        "")
  (setq-local comment-start-skip "\\(?://+\\|/\\*+\\)[[:space:]]*")
  (setq-local comment-multi-line t)

  ;; Font lock
  (setq-local font-lock-defaults
              '(meshlang-font-lock-keywords
                nil    ; highlight strings and comments too
                nil    ; case-sensitive
                nil))

  ;; Indentation
  (setq-local indent-line-function    #'meshlang-indent-line)
  (setq-local electric-indent-chars
              (append "{}" electric-indent-chars))

  ;; Misc
  (setq-local require-final-newline t)
  (setq-local tab-width meshlang-indent-offset))

;;;###autoload
(add-to-list 'auto-mode-alist '("\\.mlg\\'" . meshlang-mode))

(provide 'meshlang)
;;; meshlang.el ends here
