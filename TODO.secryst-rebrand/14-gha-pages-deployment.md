# 14 — Website deployment via GitHub Actions (not branch source)

The branch-source Pages site was disabled deliberately: deploy through a
GHA workflow (actions/deploy-pages) instead.

Steps:
1. Enable Pages with build_type=workflow (API; no branch source).
2. .github/workflows/deploy.yml: checkout -> configure-pages ->
   upload-pages-artifact(path: .) -> deploy-pages, on push to main +
   workflow_dispatch; permissions pages:write, id-token:write,
   contents:read; environment github-pages.
3. Docs-site finishing: 404.html (on-brand), robots.txt, sitemap.xml
   (www.secryst.org URLs).
4. Push, verify the workflow run goes green and all five pages serve
   the REDESIGNED content on www.secryst.org.

Acceptance: deploy workflow green; every page 200 with new content;
secryst.github.io redirects to the custom domain.
Status: DONE (workflow green, all pages live)
