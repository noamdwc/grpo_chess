# Build / Export Instructions

This repository currently has `npx` and `pandoc` available, but no local `marp` binary.

## Recommended (Marp via npx)

From repo root:

```bash
npx @marp-team/marp-cli interview_presentation/deck.md -o interview_presentation/build/deck.html
npx @marp-team/marp-cli interview_presentation/deck.md --pdf -o interview_presentation/build/deck.pdf
```

## Optional (if `marp` is installed globally)

```bash
marp interview_presentation/deck.md -o interview_presentation/build/deck.html
marp interview_presentation/deck.md --pdf -o interview_presentation/build/deck.pdf
```

## Notes

- Deck image references are relative to `interview_presentation/assets/`.
- If export fails in a restricted network environment, run the same commands on a machine with npm package download access.
