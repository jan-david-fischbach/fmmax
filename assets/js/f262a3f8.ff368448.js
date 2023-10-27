"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[5685],{3905:(e,t,r)=>{r.d(t,{Zo:()=>u,kt:()=>d});var n=r(7294);function a(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function l(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var n=Object.getOwnPropertySymbols(e);t&&(n=n.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,n)}return r}function o(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?l(Object(r),!0).forEach((function(t){a(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):l(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function i(e,t){if(null==e)return{};var r,n,a=function(e,t){if(null==e)return{};var r,n,a={},l=Object.keys(e);for(n=0;n<l.length;n++)r=l[n],t.indexOf(r)>=0||(a[r]=e[r]);return a}(e,t);if(Object.getOwnPropertySymbols){var l=Object.getOwnPropertySymbols(e);for(n=0;n<l.length;n++)r=l[n],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(a[r]=e[r])}return a}var s=n.createContext({}),f=function(e){var t=n.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):o(o({},t),e)),r},u=function(e){var t=f(e.components);return n.createElement(s.Provider,{value:t},e.children)},c="mdxType",v={inlineCode:"code",wrapper:function(e){var t=e.children;return n.createElement(n.Fragment,{},t)}},p=n.forwardRef((function(e,t){var r=e.components,a=e.mdxType,l=e.originalType,s=e.parentName,u=i(e,["components","mdxType","originalType","parentName"]),c=f(r),p=a,d=c["".concat(s,".").concat(p)]||c[p]||v[p]||l;return r?n.createElement(d,o(o({ref:t},u),{},{components:r})):n.createElement(d,o({ref:t},u))}));function d(e,t){var r=arguments,a=t&&t.mdxType;if("string"==typeof e||a){var l=r.length,o=new Array(l);o[0]=p;var i={};for(var s in t)hasOwnProperty.call(t,s)&&(i[s]=t[s]);i.originalType=e,i[c]="string"==typeof e?e:a,o[1]=i;for(var f=2;f<l;f++)o[f]=r[f];return n.createElement.apply(null,o)}return n.createElement.apply(null,r)}p.displayName="MDXCreateElement"},9003:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>o,default:()=>v,frontMatter:()=>l,metadata:()=>i,toc:()=>f});var n=r(7462),a=(r(7294),r(3905));const l={id:"farfield.unflatten_transverse_wavevectors"},o=void 0,i={unversionedId:"API/farfield.unflatten_transverse_wavevectors",id:"API/farfield.unflatten_transverse_wavevectors",title:"farfield.unflatten_transverse_wavevectors",description:"farfield.unflattentransversewavevectors",source:"@site/docs/API/farfield.unflatten_transverse_wavevectors.md",sourceDirName:"API",slug:"/API/farfield.unflatten_transverse_wavevectors",permalink:"/fmmax/API/farfield.unflatten_transverse_wavevectors",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/farfield.unflatten_transverse_wavevectors.md",tags:[],version:"current",frontMatter:{id:"farfield.unflatten_transverse_wavevectors"},sidebar:"APISidebar",previous:{title:"farfield.unflatten_flux",permalink:"/fmmax/API/farfield.unflatten_flux"},next:{title:"fft.fft",permalink:"/fmmax/API/fft.fft"}},s={},f=[{value:"<code>farfield.unflatten_transverse_wavevectors</code>",id:"farfieldunflatten_transverse_wavevectors",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],u={toc:f},c="wrapper";function v(e){let{components:t,...r}=e;return(0,a.kt)(c,(0,n.Z)({},u,r,{components:t,mdxType:"MDXLayout"}),(0,a.kt)("h3",{id:"farfieldunflatten_transverse_wavevectors"},(0,a.kt)("inlineCode",{parentName:"h3"},"farfield.unflatten_transverse_wavevectors")),(0,a.kt)("p",null,"Unflattens transverse wavevectors for a given expansion and Brillouin integration scheme."),(0,a.kt)("h4",{id:"args"},"Args:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"transverse_wavevectors"),": The transverse wavevectors array, with shape\n",(0,a.kt)("inlineCode",{parentName:"li"},"(..., num_bz_kx, num_bz_ky, ..., num_terms, 2)"),"."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"expansion"),": The expansion used for the flux."),(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"brillouin_grid_axes"),": The axes associated with the Brillouin zone grid.")),(0,a.kt)("h4",{id:"returns"},"Returns:"),(0,a.kt)("ul",null,(0,a.kt)("li",{parentName:"ul"},(0,a.kt)("strong",{parentName:"li"},"None"),": The unflattened wavevectors, with shape ",(0,a.kt)("inlineCode",{parentName:"li"},"(..., num_kx, num_ky, 2)"),".")))}v.isMDXComponent=!0}}]);