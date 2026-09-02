/**
 * Resolve a path under the app's `public/` root to an absolute URL.
 *
 * The app is built with a relative base (`base: "./"`) so the bundle works from
 * any subpath — a Hugging Face Space, GitHub Pages, a plain folder. That makes
 * `import.meta.env.BASE_URL` the string `"./"`, which a dynamic `import()` or a
 * `fetch()` inside a bundled chunk resolves against *the chunk's* URL
 * (`/assets/index-….js`) rather than the page. Resolving against
 * `document.baseURI` instead pins everything to the document, which is where
 * `public/` actually lands.
 */
export const assetUrl = (path: string): string => new URL(path, document.baseURI).href;
