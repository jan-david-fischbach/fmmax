"use strict";(self.webpackChunkdocs=self.webpackChunkdocs||[]).push([[3360],{3905:(e,t,r)=>{r.d(t,{Zo:()=>p,kt:()=>f});var i=r(7294);function n(e,t,r){return t in e?Object.defineProperty(e,t,{value:r,enumerable:!0,configurable:!0,writable:!0}):e[t]=r,e}function a(e,t){var r=Object.keys(e);if(Object.getOwnPropertySymbols){var i=Object.getOwnPropertySymbols(e);t&&(i=i.filter((function(t){return Object.getOwnPropertyDescriptor(e,t).enumerable}))),r.push.apply(r,i)}return r}function o(e){for(var t=1;t<arguments.length;t++){var r=null!=arguments[t]?arguments[t]:{};t%2?a(Object(r),!0).forEach((function(t){n(e,t,r[t])})):Object.getOwnPropertyDescriptors?Object.defineProperties(e,Object.getOwnPropertyDescriptors(r)):a(Object(r)).forEach((function(t){Object.defineProperty(e,t,Object.getOwnPropertyDescriptor(r,t))}))}return e}function m(e,t){if(null==e)return{};var r,i,n=function(e,t){if(null==e)return{};var r,i,n={},a=Object.keys(e);for(i=0;i<a.length;i++)r=a[i],t.indexOf(r)>=0||(n[r]=e[r]);return n}(e,t);if(Object.getOwnPropertySymbols){var a=Object.getOwnPropertySymbols(e);for(i=0;i<a.length;i++)r=a[i],t.indexOf(r)>=0||Object.prototype.propertyIsEnumerable.call(e,r)&&(n[r]=e[r])}return n}var s=i.createContext({}),c=function(e){var t=i.useContext(s),r=t;return e&&(r="function"==typeof e?e(t):o(o({},t),e)),r},p=function(e){var t=c(e.components);return i.createElement(s.Provider,{value:t},e.children)},l="mdxType",v={inlineCode:"code",wrapper:function(e){var t=e.children;return i.createElement(i.Fragment,{},t)}},u=i.forwardRef((function(e,t){var r=e.components,n=e.mdxType,a=e.originalType,s=e.parentName,p=m(e,["components","mdxType","originalType","parentName"]),l=c(r),u=n,f=l["".concat(s,".").concat(u)]||l[u]||v[u]||a;return r?i.createElement(f,o(o({ref:t},p),{},{components:r})):i.createElement(f,o({ref:t},p))}));function f(e,t){var r=arguments,n=t&&t.mdxType;if("string"==typeof e||n){var a=r.length,o=new Array(a);o[0]=u;var m={};for(var s in t)hasOwnProperty.call(t,s)&&(m[s]=t[s]);m.originalType=e,m[l]="string"==typeof e?e:n,o[1]=m;for(var c=2;c<a;c++)o[c]=r[c];return i.createElement.apply(null,o)}return i.createElement.apply(null,r)}u.displayName="MDXCreateElement"},7956:(e,t,r)=>{r.r(t),r.d(t,{assets:()=>s,contentTitle:()=>o,default:()=>v,frontMatter:()=>a,metadata:()=>m,toc:()=>c});var i=r(7462),n=(r(7294),r(3905));const a={id:"fmm_matrices.transverse_permittivity_vector_anisotropic"},o=void 0,m={unversionedId:"API/fmm_matrices.transverse_permittivity_vector_anisotropic",id:"API/fmm_matrices.transverse_permittivity_vector_anisotropic",title:"fmm_matrices.transverse_permittivity_vector_anisotropic",description:"fmmmatrices.transversepermittivityvectoranisotropic",source:"@site/docs/API/fmm_matrices.transverse_permittivity_vector_anisotropic.md",sourceDirName:"API",slug:"/API/fmm_matrices.transverse_permittivity_vector_anisotropic",permalink:"/fmmax/API/fmm_matrices.transverse_permittivity_vector_anisotropic",draft:!1,editUrl:"https://github.com/facebookresearch/fmmax/docs/API/fmm_matrices.transverse_permittivity_vector_anisotropic.md",tags:[],version:"current",frontMatter:{id:"fmm_matrices.transverse_permittivity_vector_anisotropic"},sidebar:"APISidebar",previous:{title:"fmm_matrices.transverse_permittivity_vector",permalink:"/fmmax/API/fmm_matrices.transverse_permittivity_vector"},next:{title:"pml.PMLParams",permalink:"/fmmax/API/pml.PMLParams"}},s={},c=[{value:"<code>fmm_matrices.transverse_permittivity_vector_anisotropic</code>",id:"fmm_matricestransverse_permittivity_vector_anisotropic",level:3},{value:"Args:",id:"args",level:4},{value:"Returns:",id:"returns",level:4}],p={toc:c},l="wrapper";function v(e){let{components:t,...r}=e;return(0,n.kt)(l,(0,i.Z)({},p,r,{components:t,mdxType:"MDXLayout"}),(0,n.kt)("h3",{id:"fmm_matricestransverse_permittivity_vector_anisotropic"},(0,n.kt)("inlineCode",{parentName:"h3"},"fmm_matrices.transverse_permittivity_vector_anisotropic")),(0,n.kt)("p",null,"Compute the transverse permittivity matrix with a vector scheme."),(0,n.kt)("p",null,"The transverse permittivity matrix E relates the electric and electric displacement\nfields, such that"),(0,n.kt)("pre",null,(0,n.kt)("code",{parentName:"pre"},"[-Dy, Dx]^T = E [-Ey, Ex]^T\n")),(0,n.kt)("h4",{id:"args"},"Args:"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"permittivity_xx"),": The xx-component of the permittivity tensor, with\nshape ",(0,n.kt)("inlineCode",{parentName:"li"},"(..., nx, ny)"),"."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"permittivity_xy"),": The xy-component of the permittivity tensor."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"permittivity_yx"),": The yx-component of the permittivity tensor."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"permittivity_yy"),": The yy-component of the permittivity tensor."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"tx"),": The x-component of the tangent vector field."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"ty"),": The y-component of the tangent vector field."),(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"expansion"),": The field expansion to be used.")),(0,n.kt)("h4",{id:"returns"},"Returns:"),(0,n.kt)("ul",null,(0,n.kt)("li",{parentName:"ul"},(0,n.kt)("strong",{parentName:"li"},"None"),": The transverse permittivity matrix.")))}v.isMDXComponent=!0}}]);