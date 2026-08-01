function normalize(s: string): string {
  return s.trim().toLowerCase().replace(/\s+/g, " ");
}

function levenshtein(a: string, b: string): number {
  const m = a.length;
  const n = b.length;
  if (m === 0) return n;
  if (n === 0) return m;

  let prev = new Array<number>(n + 1);
  let curr = new Array<number>(n + 1);
  for (let j = 0; j <= n; j++) prev[j] = j;

  for (let i = 1; i <= m; i++) {
    curr[0] = i;
    for (let j = 1; j <= n; j++) {
      const cost = a[i - 1] === b[j - 1] ? 0 : 1;
      curr[j] = Math.min(
        prev[j] + 1, // deletion
        curr[j - 1] + 1, // insertion
        prev[j - 1] + cost // substitution
      );
    }
    [prev, curr] = [curr, prev];
  }
  return prev[n];
}

/**
 * Zero-cost string similarity: normalize (trim/lowercase/collapse whitespace),
 * fast-path exact match, else normalized Levenshtein distance over at most
 * `maxCompareChars` characters of each string (bounds worst-case O(n*m) cost).
 * Returns a 0..1 score; 1 = identical after normalization, 0 = maximally different.
 */
export function normalizedSimilarity(a: string, b: string, maxCompareChars = 2000): number {
  const normA = normalize(a).slice(0, maxCompareChars);
  const normB = normalize(b).slice(0, maxCompareChars);
  if (normA === normB) return 1;

  const maxLen = Math.max(normA.length, normB.length);
  if (maxLen === 0) return 1;

  const distance = levenshtein(normA, normB);
  return Math.max(0, 1 - distance / maxLen);
}
