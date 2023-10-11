what is the resource (kind) that is going to create?
* find the resources (kind) command
  * kubectl api-resources
* find the last api version the cluster supports for kind (api-versions) 
  * kubectl api-version
* give a name in metadata (mininum)
* dive into the space of that `kind`
  * kubectl explain <kind>.spec.<subfiled>
  * kubectl explain <kind>. --resource
* browse api reference for cluster version to supplyment
* use --dry-run and --serve-dry-run for testing
* kubectl create and delete until get it right

Questions to ask before adding healthchecks
* Do we have liveness, readiness, both?
* existing HTTP endpoints that we can use?
* add new endpoints or perhaps use something else?
* healthchecks liely to use resources and/or slow down the app?
* depend on additional services?

create a config map
```
 curl -0 https://k8smastery.com/haproxy.cfg
```
