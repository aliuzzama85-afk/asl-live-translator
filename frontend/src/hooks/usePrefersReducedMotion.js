import { useEffect, useState } from "react";

const QUERY = "(prefers-reduced-motion: reduce)";

/**
 * Tracks the user's `prefers-reduced-motion` OS/browser preference live
 * (not just at mount), per PLAN.md's accessibility section: reduced-motion
 * users get frame-stepped (no intermediate lerp) skeleton playback and a
 * static loading label instead of an animated spinner/pulse.
 *
 * @returns {boolean} Whether reduced motion is currently preferred.
 */
export function usePrefersReducedMotion() {
  const [reduced, setReduced] = useState(
    () => typeof window !== "undefined" && window.matchMedia(QUERY).matches
  );

  useEffect(() => {
    const mql = window.matchMedia(QUERY);
    const onChange = (e) => setReduced(e.matches);
    mql.addEventListener("change", onChange);
    return () => mql.removeEventListener("change", onChange);
  }, []);

  return reduced;
}
